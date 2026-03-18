from __future__ import annotations

from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from .keypoint_labels import KEYPOINT_LABELS, CLASS_MAPPING
from .utils import load_keypoints, load_mesh, load_pcd


class KeypointNetDataset(Dataset):  # type: ignore[type-arg]

    def __init__(self, filter_classes: list[str] | None = None, use_texture: bool = False) -> None:
        self.keypoints = load_keypoints()
        self.samples = pd.read_csv("kp_utils/data/benchmark_indices.csv", dtype=str)
        if filter_classes is not None:
            # compute class ids from class names
            filter_classes = [list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(c)] for c in filter_classes]
            self.samples = self.samples[self.samples['class_id'].isin(filter_classes)]
        self.use_texture = use_texture

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int | slice) -> Any:
        if not isinstance(idx, int):
            if isinstance(idx, slice):
                return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
            else:
                return [self[i] for i in idx]
        class_id, mesh_id = self.get_class_and_mesh_id(idx)
        mesh = load_mesh(class_id, mesh_id, use_texture=self.use_texture)
        keypoints = self.keypoints[class_id][mesh_id]
        for kp in keypoints:
            kp['label'] = KEYPOINT_LABELS[CLASS_MAPPING[class_id]][kp['semantic_id']]
        pcd = load_pcd(class_id, mesh_id)
        return mesh, keypoints, CLASS_MAPPING[class_id], mesh_id, pcd

    def get_class_and_mesh_id(self, idx: int) -> pd.Series:  # type: ignore[type-arg]
        """
        Returns the class ID and mesh ID for the given index.
        """
        return self.samples.iloc[idx]

    @staticmethod
    def collate_fn(batch: list[Any]) -> list[Any]:
        """
        Collates a batch of meshes into a single mesh.
        """
        return batch
