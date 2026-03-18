"""I/O handler for the Schelling point dataset."""

import os
from pathlib import Path
from typing import Any, Iterator

import torch
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds
from kp_utils import SchellingDataset



class SchellingIO(IO):
    """I/O handler for saving and loading keypoints on the Schelling point dataset.

    Attributes:
        output_dir: Root directory for saved mesh and keypoint files.
    """

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        super(SchellingIO, self).__init__()

    def loop_over_test_datasets(self) -> Iterator[Any]:
        """Yield all samples from the Schelling point dataset."""
        schelling_dataset: Any = SchellingDataset()
        # schelling_dataset.transform_off_files_to_ply(self.save_mesh) # Just once when changing shapes
        yield from schelling_dataset

    def save_kps(self, mesh: Meshes, kps_3d: Pointclouds, class_title: str, mesh_id: str, semantic_id: str, postfix: str = 'keypts') -> None:
        """Save a keypoint point cloud alongside its mesh as PLY files."""
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title: str, mesh_id: str, postfix: str = 'keypts') -> dict[str, torch.Tensor]:
        """Load keypoint point clouds for a mesh, keyed by semantic ID."""
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed()
               for fname in save_dir.glob(f"{mesh_id}_*_keypts.ply")}
        return kps
