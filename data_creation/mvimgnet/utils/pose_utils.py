import os
import sys
from pathlib import Path, PurePosixPath

import torch
import imageio
import numpy as np
import skimage.transform

# Get the dir of this file
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

import colmap_read_model as read_model
from colmap_wrapper import run_colmap

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)
def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def read_meta(in_dir, use_ndc):
    """
    Read the poses_bounds.npy file produced by LLFF imgs2poses.py.
    This function is modified from https://github.com/kwea123/nerf_pl.
    """
    poses_bounds = np.load(os.path.join(in_dir, 'poses_bounds.npy'))  # (N_images, 17)

    c2ws = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    bounds = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = c2ws[0, :, -1]

    # correct c2ws: original c2ws has rotation in form "down right back", change to "right up back".
    # See https://github.com/bmild/nerf/issues/34
    c2ws = np.concatenate([c2ws[..., 1:2], -c2ws[..., :1], c2ws[..., 2:4]], -1)

    # (N_images, 3, 4), (4, 4)
    c2ws, pose_avg = center_poses(c2ws)  # pose_avg @ c2ws -> centred c2ws

    if use_ndc:
        # correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        c2ws[..., 3] /= scale_factor
    
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    results = {
        'c2ws': c2ws,       # (N, 4, 4) np
        'bounds': bounds,   # (N_images, 2) np
        'H': int(H),        # scalar
        'W': int(W),        # scalar
        'focal': focal,     # scalar
        'pose_avg': pose_avg,  # (4, 4) np
    }
    return results


def load_colmap_data(realdir):
    #
    # Get the number of cameras, their focal length, width, and height values
    #
    camerasfile = realdir / PurePosixPath('sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)

    # cam = camdata[camdata.keys()[0]]
    list_of_keys = list(camdata.keys())
    cam = camdata[list_of_keys[0]]
    print('Cameras', len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    # Read the image data
    imagesfile = realdir / PurePosixPath('sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)

    w2c_mats = []
    keypoints_pix_coords = []
    bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    ids = [imdata[k].id for k in imdata]
    print('Images #', len(names))
    perm = np.argsort(names)
    
    # We can also get the sparse points pixel positions and return them as an array of dict
    for k in imdata:
        kp_pix_coords = {}

        im = imdata[k]
        R = im.qvec2rotmat()  # Rotation matrix
        t = im.tvec.reshape([3, 1])  # Translation vector
        # Camera matrix 4x4
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
        
        assert len(im.xys) == len(im.point3D_ids)
        for i, point_id in enumerate(im.point3D_ids):
            kp_pix_coords[point_id] = im.xys[i] 
        keypoints_pix_coords.append(kp_pix_coords)
        

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    # Remove the homogenous row (last row in the 4x4 camera matirx)
    poses = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    # append the height, width, and the focal length to the camera matrix
    poses = np.concatenate(
        [poses, np.tile(hwf[..., np.newaxis], [1, 1, poses.shape[-1]])], 1)

    points3dfile = realdir / PurePosixPath('sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], - poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    return poses, pts3d, keypoints_pix_coords, perm, names, ids


def save_poses(basedir, poses, pts3d, perm, ids):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in map(ids.index, pts3d[k].image_ids):
            if len(cams) <= ind:
                raise RuntimeError('ERROR: the correct camera poses for current points cannot be accessed')
            cams[ind] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]
                                                         ) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    save_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )

        save_arr.append(np.concatenate(
            [poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)

    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)


def minify_v0(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    def downsample(imgs, f):
        sh = list(imgs.shape)
        sh = sh[:-3] + [sh[-3]//f, f, sh[-2]//f, f, sh[-1]]
        imgs = np.reshape(imgs, sh)
        imgs = np.mean(imgs, (-2, -4))
        return imgs

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgs = np.stack([imageio.imread(img)/255. for img in imgs], 0)

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        print('Minifying', r, basedir)

        if isinstance(r, int):
            imgs_down = downsample(imgs, r)
        else:
            imgs_down = skimage.transform.resize(imgs, [imgs.shape[0], r[0], r[1], imgs.shape[-1]],
                                                 order=1, mode='constant', cval=0, clip=True, preserve_range=False,
                                                 anti_aliasing=True, anti_aliasing_sigma=None)

        os.makedirs(imgdir)
        for i in range(imgs_down.shape[0]):
            imageio.imwrite(os.path.join(imgdir, 'image{:03d}.png'.format(
                i)), (255*imgs_down[i]).astype(np.uint8))


def minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(int(100./r))
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(
        imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    if not load_imgs:
        return poses, bds

    # imgs = [imageio.imread(f, ignoregamma=True)[...,:3]/255. for f in imgfiles]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def gen_poses(basezip, match_type, factors=None):

    files_needed = {'{}.bin'.format(f) for f in ('cameras', 'images', 'points3D')}
    if (sparsezip := basezip / PurePosixPath('sparse/0')).exists():
        files_had = {a.name for a in sparsezip.iterdir()}
    else:
        files_had = []
    if not files_needed.issubset(files_had):
        print('Need to run COLMAP')
        raise NotImplementedError('Need to run COLMAP')
        run_colmap(basezip, match_type)
    else:
        print('Don\'t need to run COLMAP')

    print('Post-colmap')

    poses, pts3d, _, perm, _, ids = load_colmap_data(basezip)
    basedir = Path(os.path.dirname(basezip.root.filename)) / PurePosixPath(basezip.at)
    save_poses(basedir, poses, pts3d, perm, ids)

    if factors is not None:
        print('Factors:', factors)
        minify(basedir, factors)

    print('Done with imgs2poses')

    return True


if __name__ == '__main__':
    debug_shape_path = '/home/abdeas0a/research/datasets/MVImgNet/MVImgNet_by_categories/11/16009258'
    debug_shape_point_cloud = '/datawaha/cggroup/abdeas0a/datasets/MVImgNet/mvpnet/MVPNet/11/16009258.pcd'
    poses, pts3d, perm = load_colmap_data(debug_shape_path)
