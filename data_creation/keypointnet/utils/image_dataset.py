import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class KeypointNetImageDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()

        self.data_root = data_root
        self.image_files = sorted([f for f in os.listdir(os.path.join(data_root, 'images')) if os.path.isfile(
            os.path.join(data_root, 'images', f))], key=lambda x: int(x.split('.')[0]))
        self.img_names = [f.split('.')[0] for f in self.image_files]

        # Read the visible points and their pixel positions for each rendered image
        self.max_n_points = 0
        self.visible_points = []
        self.visible_point_ids = []
        self.visible_point_pixel_pos = []

        for img_name in self.img_names:
            img_id = int(img_name)
            img_id_str = (3 - len(str(img_id))) * '0' + str(img_id)
            self.visible_points.append(np.load(os.path.join(
                data_root, 'images_data', f'{img_id_str}_visible_point_xyz.npy')))
            self.visible_point_ids.append(np.load(os.path.join(
                data_root, 'images_data', f'{img_id_str}_visible_point_ids.npy')))
            self.visible_point_pixel_pos.append(np.load(os.path.join(
                data_root, 'images_data', f'{img_id_str}_visible_point_pixel_pos.npy')))
            self.max_n_points = max(self.max_n_points, max(
                self.visible_point_ids[-1]) + 1)

        self.points = np.zeros((self.max_n_points, 3))
        for i, visible_points in enumerate(self.visible_points):
            for j, point_id in enumerate(self.visible_point_ids[i]):
                self.points[point_id] = visible_points[j]


    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_root, 'images', self.image_files[idx])
        img = Image.open(img_path).convert('RGB')

        visible_point_pixel_pos = self.visible_point_pixel_pos[idx]
        visible_point_ids = self.visible_point_ids[idx]

        points = torch.ones(self.max_n_points, 2) * -1  # Between 0 and 1
        mask = torch.zeros(self.max_n_points)
        for i, point_id in enumerate(visible_point_ids):
            points[point_id, 0] = torch.tensor(
                visible_point_pixel_pos[i][0]) * 1.0  # For the row
            points[point_id, 1] = torch.tensor(
                visible_point_pixel_pos[i][1]) * 1.0  # For the col
            mask[point_id] = 1  # 1 if the point is visible, 0 otherwise
                
        sample = {
            'img': img,
            "img_name": self.img_names[idx],
            'points_pixel_pos': points,
            'n_points': self.max_n_points,
            'visible_points_mask': mask,
        }

        return sample

    def __len__(self):
        return len(self.image_files)

    def save_points(self, path):
        np.save(path, self.points)
