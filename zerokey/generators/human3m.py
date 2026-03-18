"""Generator and I/O for Human3.6M dataset keypoint evaluation."""

from pathlib import Path
from typing import Any, Iterator

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import RotateAxisAngle
import torch
from zerokey._defaults import HUMAN3M_DATA_PATH
from zerokey.models import Molmo
from kp_utils.data.utils import load_mesh
from zerokey.io.kpnet import KPNetIO
from zerokey.generators.kpnet import KPNetGenerator

class Human3MIO(KPNetIO):
    """I/O handler for Human3.6M dataset.

    Loads human body meshes from HUMAN3M_DATA_PATH and applies a 90-degree
    Z-axis rotation to align coordinate systems.

    Attributes:
        colmap_data_path: Path to Human3M data directory.
        transform: 90-degree Z-axis rotation applied to meshes.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        colmap_data_path = HUMAN3M_DATA_PATH
        if not colmap_data_path:
            raise ValueError("HUMAN3M_DATA_PATH is not set")
        self.colmap_data_path = Path(colmap_data_path)
        self.transform = RotateAxisAngle(angle=90, axis='Z', degrees=True)

    def loop_over_test_datasets(self, use_texture: bool = False) -> Iterator[tuple[Meshes, Any, str, str, Pointclouds]]:
        """Yield a single human body mesh with predefined keypoint names."""
        scene_path = Path(self.colmap_data_path)
        mesh_id = scene_path.stem
        mesh = load_mesh(str(scene_path.absolute()), 'Scan', use_texture=False, use_normals=True, mesh_type='.obj')
        mesh = mesh.update_padded(self.transform.transform_points(mesh.verts_padded()))
        pcd = Pointclouds(*sample_points_from_meshes(mesh, return_normals=True))
        keypoints = ['knee', 'head', 'hand', 'ankle', 'foot']
        yield mesh, keypoints, 'human3m', mesh_id, pcd

    def get_kp_names_from_lable(self, class_title: str, mesh_id: str, keypoints: Any) -> dict[Any, Any]:
        """Return keypoint names indexed by ordinal position."""
        return {idx: kp for idx, kp in enumerate(keypoints)}



class Human3MGenerator(KPNetGenerator[Human3MIO, Molmo]):
    """ZeroKey generator for human body keypoint detection on Human3.6M meshes.

    Uses a larger camera distance (1.8) and finer viewpoint partition (4)
    than the base generator to accommodate full-body scans.
    """

    def sample_view_points(self, dist: float, partition: int | None = None) -> Any:
        """Sample viewpoints at distance 1.8 with partition=4."""
        return super().sample_view_points(1.8, partition=4)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vis = True

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = slice(None)) -> dict[frozenset[Any], Pointclouds]:
        """Delegate to super with all prompts processed at once."""
        return super().process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx=prompt_idx)

