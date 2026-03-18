import os
import json
import random
import trimesh
import colorsys
import itertools
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from mvimgnet.utils.image_dataset import MVImgNetImageDataset
from keypointnet.utils.image_dataset import KeypointNetImageDataset


def get_split_shapes(split, shapes_dict):
    # Get split shapes
    if split == 'train':
        split_name = 'trn'
        all_shapes = shapes_dict['train']
    elif split == 'val':
        split_name = 'val'
        all_shapes = shapes_dict['val']
    elif split == 'test':
        split_name = 'test'
        all_shapes = shapes_dict['test']

    return split_name, sorted(all_shapes)


def list_directories(path):
    """
    List all directories at depth 1 within the specified path.

    Parameters:
    path (str): The path to the directory where to look for subdirectories.

    Returns:
    list: A list of directories found at depth 1 within the specified path.
    """
    # Check if the provided path is indeed a directory
    if not os.path.isdir(path):
        raise ValueError(f"Error: The path {path} is not a valid directory.")
        return []

    # List all entries in the directory specified by path
    directory_list = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            directory_list.append(entry)

    return directory_list


def place_mark_on_pixels(image, pixel_positions, mark_color=(255, 0, 0), mark_size=2):
    """
    Place a small mark on specified pixel positions in the image.

    Args:
    - image: PIL Image object
    - pixel_positions: List of tuples containing (y, x) pixel positions
    - mark_color: Tuple representing RGB color of the mark (default is red)
    - mark_size: Integer specifying the size of the mark (default is 3 pixels)

    Returns:
    - Modified PIL Image object with marks placed on specified pixel positions
    """
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    for position in pixel_positions:
        y, x = position
        draw.ellipse([x-mark_size, y-mark_size, x+mark_size,
                     y+mark_size], fill=mark_color)
    del draw
    return marked_image


def place_mark_on_pixels_with_numbers(image, pixel_positions, mark_color=(255, 0, 0), mark_size=2, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', font_size=10):
    """
    Place a small mark and a number on specified pixel positions in the image.

    Args:
    - image: PIL Image object
    - pixel_positions: List of tuples containing (y, x) pixel positions
    - mark_color: Tuple representing RGB color of the mark (default is red)
    - mark_size: Integer specifying the size of the mark (default is 2 pixels)
    - font_path: Path to the font file
    - font_size: Size of the font

    Returns:
    - Modified PIL Image object with marks and numbers placed on specified pixel positions
    """
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    font = ImageFont.truetype(font_path, font_size)  # Load the font

    for index, position in enumerate(pixel_positions):
        y, x = position
        # Draw the ellipse
        draw.ellipse([x-mark_size, y-mark_size, x+mark_size,
                      y+mark_size], fill=mark_color)
        # Draw the number slightly offset from the ellipse
        draw.text((x + mark_size + 2, y - mark_size // 2),
                  str(index + 1), fill=mark_color, font=font)

    del draw
    return marked_image


def place_mark_on_pixels_with_different_colors(image, pixel_positions, mark_size=4, font_size=20):
    """
    Place a small mark and a number on specified pixel positions in the image using different colors for each.
    Uses the DejaVu Sans font and generates visually comfortable colors automatically.

    Args:
    - image: PIL Image object
    - pixel_positions: List of tuples containing (y, x) pixel positions
    - mark_size: Integer specifying the size of the mark (default is 2 pixels)
    - font_size: Size of the font

    Returns:
    - Modified PIL Image object with marks and numbers placed on specified pixel positions, each in a different color.
    """
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, font_size)  # Load the font

    n = len(pixel_positions)
    for index, position in enumerate(pixel_positions):
        # Generate a unique color for each mark
        hue = index / n
        # HSV to RGB conversion
        color = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
        y, x = position
        # Draw the ellipse
        draw.ellipse([x-mark_size, y-mark_size, x +
                      mark_size, y+mark_size], fill=color)
        # Draw the number slightly offset from the ellipse
        draw.text((x + mark_size + 2, y - mark_size // 2),
                  str(index + 1), fill=color, font=font)

    del draw
    return marked_image


def draw_bounding_box(org_image, bounding_box):
    """
    Draws a bounding box on the image and returns the image with the bounding box.

    Parameters:
    image_path (str): The path to the image file.
    bounding_box (list): The coordinates of the bounding box as [left, top, right, bottom].

    Returns:
    PIL.Image.Image: The image object with the bounding box drawn on it.
    """
    # Create an ImageDraw object
    image = Image.fromarray(np.array(org_image))
    draw = ImageDraw.Draw(image)

    # Draw the rectangle using the bounding box coordinates
    draw.rectangle(bounding_box, outline='red', width=3)

    # Return the modified image
    return image


def export_points(points, points_colors, obj_export_path):
    main_scene = trimesh.Scene()

    for i, point in tqdm(enumerate(points)):
        sphere = trimesh.load('../data/sphere.obj')
        assert isinstance(sphere, trimesh.Trimesh)
        scale = trimesh.transformations.scale_matrix(1)
        trans = trimesh.transformations.translation_matrix(point)
        sphere.apply_transform(scale)
        sphere.apply_transform(trans)
        sphere.visual.vertex_colors[:, :3] = points_colors[i]  # type: ignore[union-attr]
        sphere.visual.face_colors[:, :3] = points_colors[i]  # type: ignore[union-attr]
        main_scene.add_geometry(sphere)

    main_scene.export(obj_export_path)



def create_img_ann(img_name, img_width, img_height, points_pixel_coords, is_visible_point, shape_id, shape_class_id):
    ret = {}
    ret["filename"] = f"{shape_id}_{img_name}.jpg"
    ret["src_database"] = "mv_dataset"
    ret["src_annotation"] = "mv_dataset_gt"
    ret["src_image"] = "renderings"
    ret["image_width"] = img_width
    ret["image_height"] = img_height
    ret["image_depth"] = 3  # RGB
    ret["category"] = shape_class_id
    ret["pose"] = None  # not used in the telling from right training code
    ret["truncated"] = -1  # not used in the telling from right training code
    ret["occluded"] = -1  # not used in the telling from right training code
    ret["difficult"] = -1  # not used in the telling from right training code
    ret["azimuth_id"] = -1  # not used in the telling from right training code

    # Get the keypoints in the images with their semantic id
    ret["kps"] = {}

    # Get the bounding box of the object in the image
    left, right = points_pixel_coords[is_visible_point][:, 1].min(
    ), points_pixel_coords[is_visible_point][:, 1].max()
    top, bottom = points_pixel_coords[is_visible_point][:, 0].min(
    ), points_pixel_coords[is_visible_point][:, 0].max()
    ret["bndbox"] = [
        int(el) for el in [left, top, right, bottom]]

    for p in range(len(points_pixel_coords)):
        if is_visible_point[p]:
            ret["kps"][p] = [
                int(el) for el in list(points_pixel_coords[p].numpy())[::-1]]  # flip cause the original coordinates should be (col, row) and in spair dataset it is (row, col)
        else:
            ret["kps"][p] = None

    return ret


def create_image_annotations(args, shapes_dict):
    shapes_with_errors = []

    for split in args.splits:
        output_dir = os.path.join(args.save_dir, "ImageAnnotation")
        img_output_dir = os.path.join(args.save_dir, "JPEGImages")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(img_output_dir, exist_ok=True)

        # Get the split shapes
        print(split)
        _, split_shapes = get_split_shapes(split, shapes_dict)

        # Loop over the shapes and extract the per image annotations
        for _, shape_info in tqdm(enumerate(split_shapes)):
            # Read the shape id and the shape class id, the shape_info is in the following format "{shape_class_id}__{shape_id}"
            shape_class_id = shape_info.split('__')[0]
            shape_id = shape_info.split('__')[1].split('.')[0]

            # Now we can read the images and their data
            # try:
            print(f"Process {shape_id}")

            if args.dataset_name == "mvimgnet":
                shape_dataset = MVImgNetImageDataset(args.mvimgnet_dir.joinpath(shape_class_id, shape_id))
            elif args.dataset_name == "keypointnet":
                shape_dataset = KeypointNetImageDataset(os.path.join(args.save_dir, shape_class_id, shape_id))
            else:
                raise NotImplementedError

            for img_ind in range(len(shape_dataset)):
                img_raw_data = shape_dataset[img_ind]
                img = img_raw_data['img']

                img_width, img_height = img.size

                # numpy array of shape n_points x 2, -1 means this point is not visible in this image
                points_pixel_coords = img_raw_data['points_pixel_pos']
                img_name = img_raw_data['img_name'].replace('.jpg', '')
                is_visible_point = img_raw_data["visible_points_mask"].numpy().astype(
                    int).astype(bool)

                # Get the points and their pixel position in the Spair-71K dataset format
                image_data_dict = create_img_ann(img_name, img_width,
                                                    img_height, points_pixel_coords,
                                                    is_visible_point, shape_id, shape_class_id)

                os.makedirs(os.path.join(
                    output_dir, shape_class_id), exist_ok=True)
                os.makedirs(os.path.join(img_output_dir,
                            shape_class_id), exist_ok=True)

                # Save the annotation as a json file
                with open(os.path.join(output_dir, shape_class_id, f"{shape_id}_{img_name}.json"), 'w') as fout:
                    json.dump(image_data_dict, fout)

                # Save the image as a jpg
                img.save(os.path.join(img_output_dir,
                            shape_class_id, f"{shape_id}_{img_name}.jpg"))

            # except:
            #     shapes_with_errors.append((shape_class_id, shape_id))
            #     continue


def create_pair_annotation(args, shapes_dict):
    pair_id = 0

    random.seed(args.seed)
    random_state = np.random.RandomState(args.seed)
    all_sampled_combinations = []

    for split in args.splits:
        # Get the split shapes
        split_name, split_shapes = get_split_shapes(split, shapes_dict)

        output_dir = os.path.join(args.save_dir, "PairAnnotation", split_name)
        ann_output_dir = os.path.join(args.save_dir, "ImageAnnotation")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ann_output_dir, exist_ok=True)

        for _, shape_info in tqdm(enumerate(split_shapes)):
            shape_class_id = shape_info.split('__')[0]
            shape_id = shape_info.split('__')[1].split('.')[0]

            # try:
            # Now we can create all possible pairs of images
            if args.dataset_name == "mvimgnet":
                shape_dataset = MVImgNetImageDataset(args.mvimgnet_dir.joinpath(shape_class_id, shape_id))
            elif args.dataset_name == "keypointnet":
                shape_dataset = KeypointNetImageDataset(os.path.join(args.save_dir, shape_class_id, shape_id))
            else:
                raise NotImplementedError

            # Create the array
            arr = np.arange(len(shape_dataset))
            # Generate all unique combinations where (x, y) == (y, x)
            unique_combinations = list(itertools.combinations(arr, 2))
            if args.num_pairs_per_shape != -1:
                # Sample N combinations (if needed)
                sampled_combinations = random.sample(
                    unique_combinations, min(args.num_pairs_per_shape, len(unique_combinations)))
            else:
                sampled_combinations = unique_combinations

            all_sampled_combinations.append(sampled_combinations)

            for (i_src, j_target) in sampled_combinations:
                src_img_name = shape_dataset.img_names[i_src].replace(
                    '.jpg', '')
                trgt_image_name = shape_dataset.img_names[j_target].replace(
                    '.jpg', '')

                # Create the pair annotation dict
                pair_ann = create_pair_ann(pair_id, ann_output_dir, src_img_name, trgt_image_name,
                                            shape_id, shape_class_id, random_state, args.max_num_sampled_points, args.max_pair_id_len)

                # Export as json file
                with open(os.path.join(output_dir, f"{pair_ann['filename']}.json"), 'w') as fout:
                    json.dump(pair_ann, fout)

                pair_id += 1
            # except:
            #     continue

    return all_sampled_combinations


def create_shape_annotation(args, shapes_dict, all_sampled_combinations):
    pair_id = 0

    for split in args.splits:
        # Get the split shapes
        split_name, split_shapes = get_split_shapes(split, shapes_dict)

        output_dir = os.path.join(args.save_dir, "ShapeAnnotation", split_name)
        ann_output_dir = os.path.join(args.save_dir, "ImageAnnotation")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ann_output_dir, exist_ok=True)

        random_state = np.random.RandomState(2024)
        # Read all the shapes
        for i, shape_info in tqdm(enumerate(split_shapes)):
            shape_class_id = shape_info.split('__')[0]
            shape_id = shape_info.split('__')[1].split('.')[0]

            # Now we can create all possible pairs of images
            try:
                if args.dataset_name == "mvimgnet":
                    shape_dataset = MVImgNetImageDataset(args.mvimgnet_dir.joinpath(shape_class_id, shape_id))
                elif args.dataset_name == "keypointnet":
                    shape_dataset = KeypointNetImageDataset(os.path.join(args.save_dir, shape_class_id, shape_id))
                else:
                    raise NotImplementedError

                # Now we should save this in one file
                all_shape_image_pairs = []
                shape_name = f"{shape_id}:{shape_class_id}"

                # Create the array
                arr = np.arange(len(shape_dataset))
                sampled_combinations = all_sampled_combinations[i]
                
                for (i_src, j_target) in sampled_combinations:
                    src_img_name = shape_dataset.img_names[i_src].replace(
                        '.jpg', '')
                    trgt_image_name = shape_dataset.img_names[j_target].replace(
                        '.jpg', '')

                    # Create the pair annotation dict
                    pair_ann = create_pair_ann(pair_id, ann_output_dir, src_img_name, trgt_image_name,
                                               shape_id, shape_class_id, random_state, args.max_num_sampled_points, args.max_pair_id_len)
                    all_shape_image_pairs.append(pair_ann)

                    pair_id += 1

                with open(os.path.join(output_dir, f"{shape_name}.json"), 'w') as fout:
                    json.dump(all_shape_image_pairs, fout)
            except:
                continue


def sample_index_pairs(n, k, state):
    if k > n * (n - 1) // 2:
        k = min(k, n * (n - 1) // 2)

    seen_pairs = set()
    while len(seen_pairs) < k:
        i = state.randint(0, n-1)
        j = state.randint(0, n-1)
        if i != j:
            # This sorting ensures (i, j) is treated the same as (j, i)
            pair = tuple(sorted((i, j)))
            seen_pairs.add(pair)

    return list(seen_pairs)


def create_pair_ann(pair_id, ann_output_dir, src_img_name, trgt_image_name, shape_id, shape_class_id, random_state, max_num_sampled_points, max_pair_id_len):
    with open(os.path.join(ann_output_dir, shape_class_id, f"{shape_id}_{src_img_name}.json")) as fin:
        src_ann = json.load(fin)

    with open(os.path.join(ann_output_dir, shape_class_id, f"{shape_id}_{trgt_image_name}.json")) as fin:
        target_ann = json.load(fin)

    pair_name = (max_pair_id_len - len(str(pair_id))) * '0' + str(pair_id)
    pair_name = f"{pair_name}-{shape_id}-{shape_id}:{shape_class_id}"

    ret = {}

    ret["pair_id"] = pair_id
    ret["filename"] = pair_name

    ret["src_imname"] = f"{shape_id}_{src_img_name}.jpg"
    ret["trg_imname"] = f"{shape_id}_{trgt_image_name}.jpg"

    ret["src_imsize"] = [
        src_ann['image_width'], src_ann['image_height'], 3]
    ret["trg_imsize"] = [
        target_ann['image_width'], target_ann['image_height'], 3]

    ret["src_bndbox"] = src_ann["bndbox"]
    ret["trg_bndbox"] = target_ann["bndbox"]

    ret["category"] = shape_class_id
    ret["src_pose"] = 'none'  # not used in the telling from right training code
    ret["trg_pose"] = 'none'  # not used in the telling from right training code

    # Determine the common keypoints (intersection)
    kps_ids = []

    src_kps_ids = sorted(list(src_ann["kps"].keys()))
    target_kps_ids = sorted(list(target_ann["kps"].keys()))

    for _id in src_kps_ids:
        if src_ann["kps"][_id] is not None and target_ann["kps"][_id] is not None:
            assert _id in target_kps_ids

            kps_ids.append(_id)

    # Sample only up to MAX_NUM_SAMPLED_POINTS keypoints
    choices = sorted(random_state.choice(kps_ids, size=min(
        len(kps_ids), max_num_sampled_points), replace=False))

    ret['src_kps'] = [src_ann["kps"][k] for k in choices]
    ret['trg_kps'] = [target_ann["kps"][k] for k in choices]
    ret['kps_ids'] = choices

    ret["mirror"] = None  # not used in the telling from right training code
    # not used in the telling from right training code
    ret["viewpoint_variation"] = None
    # not used in the telling from right training code
    ret["scale_variation"] = None
    ret["truncation"] = None  # not used in the telling from right training code
    ret["occlusion"] = None  # not used in the telling from right training code

    return ret
