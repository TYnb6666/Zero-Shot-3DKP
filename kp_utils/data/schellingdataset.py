from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset

from pytorch3d.io import IO
import os
import numpy as np

# TODO: Correct this
# 1 - 20 human body
# 21 - 40 container
# 41 - 60 glasses
# 61 - 80 airplane
# 81 - 100 insect (ant)
# 101 - 120 chair
# 121 - 140 spider
# 141 - 160 table
# 161 - 180 teddy bear
# 181 - 200 human hand
# 201 - 220 plier
# 221 - 240 dolphin
# 241 - 260 bird
# 261 - 280 swirl
# 281 - 300 big armadillo
# 301 - 320 human head
# 321 - 340 deformed cubes
# 341 - 360 weird cylinder
# 361 - 380 jar
# 381 - 400 animal


CLASS_MAPPING = {
    41: 'sunglasses',
    64: 'airplane',
    112: 'chair',
    302: 'human head'

}

SHAPES = [41, 64,112,302]
ALL_SHAPES = [i for i in range(1,401)]

io = IO()
OFF_MESHES_PATH = 'SchellingData/Meshes/'
MESHES_PATH = 'SchellingData/PlyMeshes/'
ANNOTATION_PATH = 'SchellingData/Distributions'
import trimesh
from tqdm import tqdm

class SchellingDataset(Dataset):  # type: ignore[type-arg]

    def __init__(self) -> None:
        pass

    def transform_off_files_to_ply(self, saving_function: Callable[..., Any]) -> None:
        from pytorch3d.renderer import TexturesVertex
        import torch

        for idx in tqdm(ALL_SHAPES):
            with open(os.path.join(OFF_MESHES_PATH, f"{idx}.off"), 'r') as _f:
                mesh = trimesh.load(_f, file_type='off')
            
            # Create ply file
            ply_file = os.path.join(MESHES_PATH, f"{idx}.ply")
            mesh.export(ply_file)
            mesh = io.load_mesh(ply_file, include_textures=False)
            verts = mesh.verts_packed()
            assert verts is not None
            mesh.textures = TexturesVertex(verts_features=torch.ones_like(verts[None]) * 0.7)
            saving_function(mesh,ply_file)

    def __len__(self) -> int:
        return len(SHAPES)

    def __getitem__(self, idx: int | slice) -> Any:
        if not isinstance(idx, int):
            if isinstance(idx, slice):
                return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
            else:
                return [self[i] for i in idx]
            
        mesh_id = SHAPES[idx]

        ply_file = os.path.join(MESHES_PATH, f"{mesh_id}.ply")
        missing_file = not os.path.exists(ply_file)
        if missing_file:
            # Load off file
            with open(os.path.join(OFF_MESHES_PATH, f"{mesh_id}.off"), 'r') as _f:
                mesh = trimesh.load(_f, file_type='off')
            # Create ply file
            mesh.export(ply_file)

        # Loading the mesh
        mesh = io.load_mesh(ply_file, include_textures=True)

        # Loading array of schelling distribution
        distribution_file = os.path.join(ANNOTATION_PATH,f"{mesh_id}.val")
        annotations = np.loadtxt(distribution_file)

        return mesh, CLASS_MAPPING[mesh_id], mesh_id, annotations

    @staticmethod
    def collate_fn(batch: list[Any]) -> list[Any]:
        """
        Collates a batch of meshes into a single mesh.
        """
        return batch