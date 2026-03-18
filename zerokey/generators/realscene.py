"""Generator and I/O for real-scene keypoint evaluation using COLMAP reconstructions."""

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
import torch
from data_creation.scene.cameras import convert_camera_from_gs_to_pytorch3d
from zerokey._defaults import COLMAP_DATA_PATH
from zerokey.models import Molmo
from kp_utils.data.utils import load_mesh
from zerokey.io.kpnet import KPNetIO
from zerokey.generators.kpnet import KPNetGenerator
from data_creation.scene.dataset_readers import sceneLoadTypeCallbacks

class RealSceneIO(KPNetIO):
    """I/O handler for COLMAP-reconstructed real scenes.

    Loads scene data from COLMAP_DATA_PATH, including camera poses,
    point clouds, and textured meshes.

    Attributes:
        scene_info: COLMAP scene metadata including cameras and point cloud.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        colmap_data_path = COLMAP_DATA_PATH
        if not colmap_data_path:
            raise ValueError("COLMAP_DATA_PATH is not set")
        colmap_data_path = Path(colmap_data_path)
        self.scene_info = sceneLoadTypeCallbacks["Colmap"](colmap_data_path, images=None, eval=False)


    def loop_over_test_datasets(self, use_texture: bool = False) -> Iterator[tuple[Meshes, Any, str, str, Pointclouds]]:
        """Yield the COLMAP scene mesh with predefined keypoint names."""
        scene_path = Path(self.scene_info.ply_path).parents[2]
        mesh_id = scene_path.stem
        mesh = load_mesh(str(scene_path.absolute()), 'refined_mesh', use_texture=True)
        pcd = self.scene_info.point_cloud
        pcd = Pointclouds(points=torch.from_numpy(pcd.points[np.newaxis]),
                          normals=torch.from_numpy(pcd.normals[np.newaxis]),
                          features=torch.from_numpy(pcd.colors[np.newaxis]))
        keypoints = ['plant container', 'table corner']
        yield mesh, keypoints, 'colmap', mesh_id, pcd

    def get_kp_names_from_lable(self, class_title: str, mesh_id: str, keypoints: Any) -> dict[Any, Any]:
        """Return keypoint names indexed by ordinal position."""
        return {idx: kp for idx, kp in enumerate(keypoints)}



class RealSceneGenerator(KPNetGenerator[RealSceneIO, Molmo]):
    """ZeroKey generator for real-scene keypoint detection using COLMAP cameras.

    Uses actual COLMAP camera poses instead of synthetic viewpoints, and
    substitutes real captured images for rendered ones.

    Attributes:
        views: Subsampled list of COLMAP camera views (every 16th frame).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, res=(668, 1200), scale=1.25, **kwargs)
        scene_info = self.io.scene_info
        self.views = scene_info.train_cameras + scene_info.test_cameras
        # random.shuffle(self.views)
        self.views = self.views[0:-1:16]
        self.vis = True

    def views_from_model(self, mesh: Meshes, views: Any, batch_size: int | None = None, device: str | torch.device = "cuda") -> tuple[torch.Tensor, Any, CamerasBase, Any]:
        """Render using COLMAP cameras and replace rendered images with real captures."""
        # R, T = zip(*[colmap_to_pytorch3d(
        #     rotation=torch.from_numpy(view.R).to(device=device),
        #     translation=torch.from_numpy(view.T).unsqueeze_(-1).to(device=device),
        #     device=device) for view in self.views])
        # FoV = [view.FovY for view in self.views]

        # cameras = FoVPerspectiveCameras(R=torch.stack(R), T=torch.stack(T), fov=FoV, degrees=False, device=device)
        cameras = convert_camera_from_gs_to_pytorch3d(self.views)
        _images, fragments, R, T = super().views_from_model(mesh, cameras, batch_size=batch_size, device=device)
        images_raw = torch.stack([torch.from_numpy(np.asanyarray(v.image.convert("RGBA"))).permute(2, 0, 1) for v in self.views])
        images_raw = images_raw.to(device=device) / 255.
        return images_raw, fragments, R, T

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = slice(None)) -> dict[frozenset[Any], Pointclouds]:
        """Delegate to super with all prompts processed at once."""
        return super().process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=prompt_idx)

