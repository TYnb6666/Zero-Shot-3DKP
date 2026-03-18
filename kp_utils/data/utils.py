from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.io.pluggable_formats import PointcloudFormatInterpreter
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Pointclouds

from zerokey._defaults import KEYPOINT_DATASET_PATH

io = IO()

KEYPOINTNET_DATASET_PATH = KEYPOINT_DATASET_PATH
MESHES_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, os.environ.get("KEYPOINT_DATASET_MESHES_PATH",
                                                                    "ShapeNetCore.v2.ply"))
PCDS_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, os.environ.get("KEYPOINT_DATASET_PCDS_PATH",
                                                                  "pcds"))
KEYPOINTS_PATH = os.path.join(KEYPOINTNET_DATASET_PATH, "annotations")


def load_mesh(class_id: str, mesh_id: str, use_texture: bool = True, use_normals: bool = False, mesh_type: str = '.ply') -> Any:
    """
    Loads the mesh with the given ID.
    """
    mesh = io.load_mesh(os.path.join(MESHES_PATH, class_id, mesh_id + mesh_type), include_textures=use_texture)
    if not use_texture:
        points = mesh.verts_packed()  # Shape: (N, 3)
        assert points is not None
        if use_normals:
            verts_features: list[torch.Tensor] | torch.Tensor = list(mesh.verts_normals_list())
        elif False:
            # Min-max normalize the points to be in range [0, 1]
            min_coords = points.min(dim=0).values
            max_coords = points.max(dim=0).values
            verts_features = (points - min_coords) / (max_coords - min_coords)
            verts_features = verts_features[None]
        else:
            verts_features = torch.ones_like(points[None]) * 0.7
        mesh.textures = TexturesVertex(verts_features=verts_features)
    return mesh


def load_pcd(class_id: str, mesh_id: str) -> Pointclouds:
    """
    Loads the point cloud for the given class and mesh ID.
    """
    pcd = io.load_pointcloud(os.path.join(PCDS_PATH, class_id, mesh_id + ".pcd"))
    return pcd


def load_keypoints() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Loads the keypoints from the annotations folder.
    """
    keypoints = {}
    for file in os.listdir(KEYPOINTS_PATH):
        with open(os.path.join(KEYPOINTS_PATH, file), "r") as f:
            labels = json.load(f)
            for l in labels:
                if l["class_id"] not in keypoints:
                    keypoints[l["class_id"]] = {}
                if l["model_id"] in keypoints[l["class_id"]]:
                    continue
                keypoints[l["class_id"]][l["model_id"]] = l["keypoints"]
    return keypoints


def naive_read_pcd(path: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=float)
    colors = np.array(data[:, -1], dtype=int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


class PCDKeypointNetInterpreter(PointcloudFormatInterpreter):
    def read(self, path: str, device: str | torch.device, path_manager: Any, **kwargs: Any) -> Pointclouds:
        pc, colors = naive_read_pcd(path)
        return Pointclouds(points=[torch.from_numpy(pc).to(device)],
                           features=[torch.from_numpy(colors).to(device)])


io.register_pointcloud_format(PCDKeypointNetInterpreter())
