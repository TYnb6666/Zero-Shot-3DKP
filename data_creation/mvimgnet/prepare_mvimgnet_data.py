import os
import json
import argparse
import zipfile
from contextlib import ExitStack
from pathlib import PurePosixPath, Path

import numpy as np
from tqdm.auto import tqdm
from pyntcloud import PyntCloud
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm.contrib import tenumerate

from data_creation.mvimgnet.utils.pcd import read_pcd
from kp_utils.pose_utils import gen_poses, load_colmap_data


def get_new_point_id(old_point_id, new_point_ids_dict, cur_id=None):
    """
    Generates a new point ID for a given old point ID, ensuring uniqueness across the point cloud.

    Parameters:
    - old_point_id (int): The old point ID that needs to be replaced or mapped to a new ID.
    - new_point_ids_dict (dict): A dictionary mapping old point IDs to their corresponding new IDs.
    - cur_id (int, optional): The current ID counter. If provided, it will be used for assigning a new ID. 
        If not provided, the function expects the `cur_id` to be passed from the calling function.

    Returns:
    tuple: A tuple containing the new_point_id and the increment.
        - new_point_id (int): The new point ID generated or retrieved for the old point ID.
        - increment (int): If a new point ID was generated, this will be 1 indicating an increment in the ID counter. 
            Otherwise, it will be 0.
    """
    inc = 0

    if old_point_id not in new_point_ids_dict:
        assert cur_id is not None and cur_id >= 0
        inc = 1
        new_point_ids_dict[int(old_point_id)] = int(cur_id)

        return new_point_ids_dict[old_point_id], inc

    return new_point_ids_dict[old_point_id], inc


def filter_pointcloud(sparse_pointcloud_dict, object_dense_pointcloud, distance_threshold=0.02):
    """
    Filters a sparse point cloud by retaining points that are within a specified
    distance threshold from any point in the 3D object point cloud. 

    This function is used to filter out the points in the sparse point cloud that doesn't belong
    to the 3D object point cloud.

    Note: we are using the sparse point cloud because it is provided already in the data where they 
    exist in each of the real images and there is no need for the rendering to take place.

    Parameters:
    - sparse_pointcloud_dict (dict): A dictionary where the keys are point IDs and
        the values are objects with a property `xyz`, representing the x, y, z coordinates
        of the points in the sparse point cloud.
    - object_dense_pointcloud (ndarray): A NumPy array containing the x, y, z coordinates
        of points in the dense point cloud of the object.
    - distance_threshold (float): The maximum distance between points in the sparse
        point cloud and the nearest point in the object point cloud for the point to be
        considered as belonging to the object.

    Returns:
    tuple: A tuple containing two items:
        - new_point_ids_dict (dict): A dictionary where the keys are new IDs assigned to the
        points that are within the distance threshold from the object, and the values
        are the original point IDs from `sparse_pointcloud_dict`.
        - point_belongs_to_the_object (defaultdict): A dictionary indicating whether a
        point (indexed by new point ID) belongs to the object. The values are boolean.

    Note:
    - The `get_new_point_id` function is assumed to handle the creation of new point IDs
        and determine if there is an increment in `cur_id`.
    """
    new_point_ids_dict = {}
    point_belongs_to_the_object = defaultdict(bool)
    cur_id = 0

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(
        object_dense_pointcloud)

    for point_id, point_data in sparse_pointcloud_dict.items():
        distances, indices = nbrs.kneighbors(np.array([point_data.xyz]))

        if distances[0] >= distance_threshold:
            continue

        new_point_id, inc = get_new_point_id(
            point_id, new_point_ids_dict, cur_id)
        cur_id += inc

        point_belongs_to_the_object[new_point_id] = True

    return new_point_ids_dict, point_belongs_to_the_object


def get_image_data(img_names, keypoints_pix_coords, new_point_ids_dict, point_belongs_to_the_object):
    images_data = defaultdict(lambda: ([], []))

    for i_img, img_data in enumerate(keypoints_pix_coords):

        for old_point_id, point_pix_coor in img_data.items():
            new_point_id = new_point_ids_dict.get(old_point_id, None)

            if new_point_id is None:
                continue

            if old_point_id == -1 or not point_belongs_to_the_object.get(new_point_id, False):
                continue

            img_id = img_names[i_img]
            images_data[img_id][0].append(new_point_id)
            images_data[img_id][1].append([int(el) for el in point_pix_coor])

    return images_data


def prepare_split(args, split, mvpnet_path, mvi_paths):
    failed_shapes = []
    distance_threshold = args.distance_threshold
    overwrite = args.overwrite

    with open(os.path.join(args.save_dir, f'{split}_shapes.json')) as fin:
        split_shapes = json.load(fin)

    for _, shape_info in tenumerate(split_shapes):
        # Read the shape id and the shape class id, the shape_info is in the following format "{shape_class_id}__{shape_id}"
        print(shape_info)
        shape_class_id = shape_info.split('__')[0]
        print(f"shape class id", shape_class_id)
        shape_id = shape_info.split('__')[1].split('.')[0]
        print(f"shape id", shape_id)
        try:
            mvi_dir, = [mvi_path for a in mvi_paths if (mvi_path := a.joinpath(shape_class_id, shape_id)).exists()]
            shape_dir = Path(os.path.dirname(mvi_dir.root.filename)) / PurePosixPath(mvi_dir.at)
            shape_point_cloud_path = mvpnet_path.joinpath('MVPNet', shape_class_id, f"{shape_id}.pcd")
            print(f"shape point cloud path", shape_point_cloud_path)
            assert shape_point_cloud_path.exists()

            # Extract the pose meta data file if not found
            if not os.path.exists(shape_dir / 'poses_bounds.npy'):
                shape_dir.mkdir(parents=True, exist_ok=True)
                gen_poses(mvi_dir, None)

            # If already processed, skip
            if not overwrite and os.path.isfile(shape_dir / 'points_pix_coordinates_v2.json'):
                continue

            print(f"Processing {shape_class_id}/{shape_id}...")

            _, sparse_pointcloud_dict, keypoints_pix_coords, perm, img_names, _ = load_colmap_data(mvi_dir)
            object_dense_pointcloud = PyntCloud(**read_pcd(shape_point_cloud_path)).xyz

            new_point_ids_dict, point_belongs_to_the_object = filter_pointcloud(
                sparse_pointcloud_dict, object_dense_pointcloud, distance_threshold)

            images_data = get_image_data(
                img_names, keypoints_pix_coords, new_point_ids_dict, point_belongs_to_the_object)

            with open(os.path.join(shape_dir, 'points_pix_coordinates_v2.json'), 'w') as fout:
                json.dump({
                    'data': images_data,
                    'permutation': [int(el) for el in list(perm)],
                    'point_ids': new_point_ids_dict
                }, fout)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break  # Exit the loop when a KeyboardInterrupt occurs
        except:
            failed_shapes.append(f"{shape_class_id}/{shape_id}")
            print(f"Error processing {shape_class_id}/{shape_id}")
            raise
    return failed_shapes


def main(args):
    failed_shapes = []

    with zipfile.ZipFile(os.path.join(args.mvpnet_dir, 'MVPNet.zip')) as mvpnet_zip, ExitStack() as stack:
        mvi_zips = [zipfile.Path(stack.enter_context(zipfile.ZipFile(mvi_file)))
                    for mvi_file in tqdm(Path(args.mvimgnet_dir).glob('mvi_*.zip'))]

        # Read the shapes
        for split in args.splits:
            failed_shapes.extend(prepare_split(args, split, zipfile.Path(mvpnet_zip), mvi_zips))

    # Now save the failed shapes 
    with open(os.path.join(args.save_dir, 'failed_shapes.json'), 'w') as fout:
        json.dump(failed_shapes, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the points and their pixel positions in each of the shape rendered images.")

    # Add arguments
    parser.add_argument('--mvimgnet_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--mvpnet_dir', type=str, required=True,
                        help='Directory containing the point cloud data (MVPNet) of the MVImgNet dataset')
    parser.add_argument('--distance_threshold', type=float, default=0.02,
                        help='Distance threshold for filtering point clouds')
    parser.add_argument('--overwrite', action='store_true',
                        help='Whether to overwrite existing data')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path of the dataset save dir. should be the same one used in sample_dataset.py')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='List of dataset splits (default: ["train", "val"])')

    # Parse arguments
    main(parser.parse_args())
