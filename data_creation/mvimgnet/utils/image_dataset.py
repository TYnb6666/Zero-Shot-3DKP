import os
import json
from pathlib import PurePosixPath, Path

import torch

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset


class MVImgNetImageDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()

        # Read the visible points and their pixel positions for each rendered image
        self.point_to_pixel_pos = defaultdict(list)
        shape_dir = Path(os.path.dirname(data_root.root.filename)) / PurePosixPath(data_root.at)
        # Read the point pixel coordinates
        with open(os.path.join(shape_dir, 'points_pix_coordinates_v2.json')) as fin:
            self.shape_data = json.load(fin)

        max_n_point_ids = len(self.shape_data["point_ids"])
        self.max_n_points = max_n_point_ids
        assert max(self.shape_data["point_ids"].values(), default=-1) + 1 == self.max_n_points

        self.data_root = shape_dir

        self.img_names = []
        self.data = {}
        for img_name, img_data in self.shape_data['data'].items():
            self.img_names.append(img_name)

            vis_points_mask = torch.zeros(self.max_n_points)
            pnts_pixel_pos = torch.ones(self.max_n_points, 2) * -1

            p_ids, p_coords = img_data
            for j, point_id in enumerate(p_ids):
                # it is y row
                pnts_pixel_pos[point_id, 0] = p_coords[j][::-1][0]
                # it is x col
                pnts_pixel_pos[point_id, 1] = p_coords[j][::-1][1]
                # 1 if the point is visible, 0 otherwise
                vis_points_mask[point_id] = True

            img = Image.open(data_root.joinpath('images', img_name).open('rb'))
            self.data[img_name] = {
                "visible_points_mask": vis_points_mask,
                "points_pixel_pos": pnts_pixel_pos,
                "img": img,
                "img_name": img_name
            }
            img._exclusive_fp = True

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        return self.data[img_name]

    def __len__(self):
        return len(self.img_names)
