import os
import json
import pandas as pd
import numpy as np
import point_cloud_utils as pcu
from pathlib import Path

from PIL import Image
from copy import deepcopy
from tqdm.auto import tqdm
from PIL import Image

from data_creation.keypointnet.utils.render_utils import *
from data_creation.keypointnet.utils.camera_utils import *
from data_creation.keypointnet.utils.mesh_utils import *

os.environ["PYOPENGL_PLATFORM"] = "egl"


def run(args, mesh_id, mesh_class, keypoint_ann, seed, n_points=1024, keypoints_only=False):
    mesh_output_dir = Path(args.save_dir, mesh_class, mesh_id)
    os.makedirs(mesh_output_dir, exist_ok=True)

    # Read Mesh
    print("\tReading the mesh...")
    mesh_path = os.path.join(args.keypointnet_dir, 'ShapeNetCore.v2.ply', mesh_class, f'{mesh_id}.ply')
    mesh, translation, scale = prepare_mesh(mesh_path, mode="sphere")
    mesh.visual.face_colors = np.array([255 // 2, 255 // 2, 255 // 2])
    print("\tdone.")

    keypoint_ann = keypoint_ann[mesh_id]
    n_keypoints = len(keypoint_ann['keypoints'])
    print(f"\tNumber of keypoints: {n_keypoints}")

    random_state = np.random.RandomState(seed)

    ###############################
    ##### Sampling point clouds ###
    ###############################
    f, v, n = np.array(deepcopy(mesh.faces)), np.array(
        deepcopy(mesh.vertices)), np.array(deepcopy(mesh.vertex_normals))
    _sampled_points_face_ids, bc = pcu.sample_mesh_poisson_disk(
        v, f, n_points * 2, random_seed=2023)
    _sampled_points = pcu.interpolate_barycentric_coords(
        f, _sampled_points_face_ids, bc, v)

    ids = random_state.choice(np.arange(len(_sampled_points)), size=min(
        len(_sampled_points), n_points - n_keypoints), replace=False)

    sampled_points = _sampled_points[ids]
    sampled_points_face_ids = _sampled_points_face_ids[ids]

    # Since we are centering the shape in a unit sphere around the origin,
    # we need to apply translation and scaling to the loaded keypoints
    tmp_xyz = np.array([el['xyz'] for el in keypoint_ann['keypoints']])
    tmp_semantic_ids = np.array([el['semantic_id']
                                 for el in keypoint_ann['keypoints']])
    tmp_face_ids = np.array([el['mesh_info']['face_index']
                             for el in keypoint_ann['keypoints']])
    tmp_xyz = tmp_xyz + translation
    tmp_xyz = tmp_xyz / scale

    ######################################
    # Use the keypoints only
    ######################################
    if keypoints_only:
        sampled_points = np.concatenate([tmp_xyz], axis=0)
        sampled_points_face_ids = np.concatenate([tmp_face_ids])
        suffix = '_kp'
    else:
        ######################################
        # Use the keypoints + the sampled points
        ######################################
        sampled_points = np.concatenate([sampled_points, tmp_xyz], axis=0)
        sampled_points_face_ids = np.concatenate(
            [sampled_points_face_ids, tmp_face_ids])
        suffix = ''
    print("\tSampling point cloud. done.")

    mesh.export(os.path.join(mesh_output_dir, "normalized_mesh.obj"))

    # Render Mesh
    camera_poses = get_camera_poses(
        target=np.mean(mesh.vertices, axis=0), seed=seed,
        add_random_views=bool(args.num_random_views), num_random_views=args.num_random_views)

    # print(camera_poses.shape)
    renderer = Render(size=args.render_res, camera_poses=camera_poses)
    print("\tRendering the mesh...")
    triangle_ids, rendered_images, normal_maps, depth_images, p_images = renderer.render(
        path=None,
        clean=None,
        intensity=6.0,  # Light intensity
        mesh=mesh,
        only_render_images=False,
        color=None,
        correct_n=True,
    )

    _, rendered_images, _, _, _ = renderer.render(
        path=None,
        clean=None,
        intensity=6.0,  # Light intensity
        mesh=mesh,
        only_render_images=True,
        color=None,
        correct_n=True,
    )
    print("\tdone.")

    # Save the images and the image features
    image_save_dir = mesh_output_dir / "images"
    os.makedirs(image_save_dir, exist_ok=True)
    for _i, _img in enumerate(rendered_images):
        i_str = str(_i)
        Image.fromarray(_img).save(image_save_dir / f"{i_str}.png")

    # Save the sampled points
    np.save(mesh_output_dir / f'sampled_points{suffix}.npy', sampled_points, allow_pickle=True)
    np.save(mesh_output_dir / f'camera_poses{suffix}.npy', camera_poses, allow_pickle=True)
    return rendered_images, mesh, sampled_points, sampled_points_face_ids, p_images, triangle_ids


def render_split_shapes(args, split_shapes_df, keypoint_ann, keypoints_only=False):
    if keypoints_only:
        suffix = '_kp'
    else:
        suffix = ''

    for i_row, row in tqdm(split_shapes_df.iterrows()):
        # try:
        # if os.path.exists(os.path.join(args.save_dir, row.class_id, row.mesh_id)):
        #     print(f"Ignoring {row.mesh_id} mesh")
        #     continue

        mesh_class_id = row.mesh_class
        mesh_id = row.mesh_id

        # Render images for the current shapes and get the sampled point cloud on the mesh surface and the face ids they belong to
        # Get also the per pixel intersection points (p_image) and the per pixel face id (triangle_ids)
        rendered_images, mesh, sampled_points, sampled_points_face_ids, p_images, triangle_ids = run(args,
                                                                                                     mesh_id,
                                                                                                     mesh_class_id,
                                                                                                     keypoint_ann,
                                                                                                     seed=i_row + args.seed,
                                                                                                     n_points=args.n_points,
                                                                                                     keypoints_only=keypoints_only)

        # Now get the face_id /pixel position/3D intersection point of each sampled point in each rendererd image if it is visible
        target_pos = project_images_to_pointcloud(
            mesh, sampled_points, sampled_points_face_ids, p_images, triangle_ids)

        target_export_dir = os.path.join(args.save_dir, str(
            row.mesh_class), str(row.mesh_id), 'images_data')
        os.makedirs(target_export_dir, exist_ok=True)

        # Now we can export the position of the points in each view (if visible)
        for i in range(len(rendered_images)):
            i_str = str(i)
            i_str = (3 - len(i_str)) * '0' + i_str

            np.save(os.path.join(
                target_export_dir, f'{i_str}_visible_point_xyz{suffix}.npy'), target_pos[i][0], allow_pickle=True)
            np.save(os.path.join(
                target_export_dir, f'{i_str}_visible_point_ids{suffix}.npy'), target_pos[i][1], allow_pickle=True)
            np.save(os.path.join(
                target_export_dir, f'{i_str}_visible_point_pixel_pos{suffix}.npy'), target_pos[i][3], allow_pickle=True)

            # not needed anymore and not saved to the storage overhead/consumption
            # np.save(os.path.join(target_export_dir, f'{i_str}_visible_point_face_ids.npy'), target_pos[i][2], allow_pickle=True)
            # np.save(os.path.join(target_export_dir, f'{i_str}_visible_point_scaled_pixel_pos.npy'), target_pos[i][4], allow_pickle=True)
            # np.save(os.path.join(target_export_dir, f'{i_str}_p_image.npy'), p_images[i], allow_pickle=True)
            # np.save(os.path.join(target_export_dir, f'{i_str}_triangle_ids.npy'), triangle_ids[i], allow_pickle=True)
        # except KeyboardInterrupt:
        #     print("\nProcess interrupted by the user.")
        #     break
        # except:
        #     print("failed shape with id:", row.mesh_id)
        #     continue


def main(args):
    # Read the keypoint ground truth annotations for all shapes of all categories
    with open(os.path.join(args.keypointnet_dir, 'annotations/all.json')) as fin:
        keypoint_ann = json.load(fin)
        keypoint_ann = {el['model_id']: el for el in keypoint_ann}

    # Now read the split shapes and render them accordingly
    # Note: the test shapes only have the keypoints and no other sample points which is not like the training and validation shapes
    for split in args.splits:

        split_shapes_df = pd.read_csv(os.path.join(
            args.save_dir, f'{split}_shapes.csv'), dtype=str)
        if split == 'test':
            # Generate a version with sampled points + keypoints 
            # and another version with only the keypoints (this version is used during the evaluation of 3D src, target pairs)
            render_split_shapes(args, split_shapes_df,
                                keypoint_ann, keypoints_only=True)
            render_split_shapes(args, split_shapes_df,
                                keypoint_ann, keypoints_only=False)

        else:
            render_split_shapes(args, split_shapes_df,
                                keypoint_ann, keypoints_only=False)


