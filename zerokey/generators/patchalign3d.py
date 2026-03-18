"""PatchAlign3D baseline generator using patch-level 3D feature extraction and CLIP matching."""

from collections import defaultdict
from typing import Any

from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
import torch
import numpy as np
from io import BytesIO

from zerokey.rendering import camera_from_eye_at_up
from patchalign3d.inference.explore_pc_patches import PatchExplorer
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO


def find_affine_transform(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find optimal affine transformation from a to b.
    Solves: b = a @ A.T + t

    Args:
        a: [N, 3] source points
        b: [N, 3] target points

    Returns:
        A: [3, 3] affine matrix
        t: [3] translation vector
    """
    # Append a column of ones to form homogeneous coordinates [x, y, z, 1],
    # so the affine transform b = a @ A.T + t can be solved as b = a_homo @ M
    # where M = [[A.T], [t]] is a [4, 3] matrix combining rotation and translation.
    N = a.shape[0]
    a_homo = torch.cat([a, torch.ones(N, 1, device=a.device, dtype=a.dtype)], dim=1)  # [N, 4]

    # Solve the overdetermined system a_homo @ M = b via least squares (lstsq).
    # M[:3, :] contains A transposed, M[3, :] contains the translation vector t.
    M = torch.linalg.lstsq(a_homo, b).solution  # [4, 3]

    A = M[:3, :].T  # [3, 3]
    t = M[3, :]     # [3]

    return A, t


def remove_stop_words(query: str, additional_stop_words: tuple[str, ...] = ()) -> str:
    """
    Remove common stop words from a query string.

    Args:
        query: String query to process

    Returns:
        String with stop words removed
    """
    # Common English stop words
    stop_words = {
        'a', 'all', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
        'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
        'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
        'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
        'who', 'oil', 'its', 'now', 'find', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part', *additional_stop_words
    }
    # Split query into words, filter out stop words, and rejoin
    return ' '.join(word for word in query.split() if word.lower() not in stop_words)


class PatchAlign3DGenerator(KPNetGenerator[KPNetIO, PatchExplorer]):
    """Base PatchAlign3D generator with generic IO type for subclass customization."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, scale=1, **kwargs)
        self.vis = False

    def sample_view_points(self, dist: float, partition: int | None = None) -> Any:
        """Sample a single viewpoint (partition=-1)."""
        return super().sample_view_points(dist, partition=-1)

    def views_from_model(self, mesh: Meshes, views: Any, batch_size: int | None = None, device: str | torch.device = "cuda") -> tuple[torch.Tensor, None, Any, None]:
        """Project mesh vertices into camera-space coordinates for each view.

        Instead of rendering images, transforms all mesh vertices into each
        camera's coordinate frame for direct patch feature extraction.

        Returns:
            Tuple of (per-view vertices [B, V, 3], None, cameras, None).
        """
        mesh = mesh.to(device=device)
        with torch.no_grad():
            verts = mesh.verts_packed()
            assert verts is not None
            target = verts.mean(dim=0, keepdim=True)
            views = torch.as_tensor(views, dtype=target.dtype, device=target.device) + target
        views = camera_from_eye_at_up(views, target, device=device)

        pts = torch.stack([view.get_world_to_view_transform().transform_points(mesh.verts_packed()) for view in views])

        return pts, None, views, None

    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp: dict[Any, Any], class_title: str, mesh_id: str) -> defaultdict[str, list[dict[str, Any]]]:
        """Extract patch features per view and match each keypoint query via CLIP text embeddings.

        Serializes per-view vertices as npz, runs PatchExplorer.extract() and match(),
        and returns per-query match results keyed by query string.
        """
        npz_files = []
        for image in tensor_images:
            with BytesIO() as in_buffer:
                np.savez(in_buffer, image.cpu().numpy())
                in_buffer.seek(0)
                npz_files.append(self.multimodal.extract(input=in_buffer, output=None))

        # Match query
        ret = defaultdict(list)
        for query, *hints in kp.values():
            key = query
            if hints:
                *hints, query = query, *hints
            if not hints:
                query, *hints = (query,)

            # Remove stop words from query
            query = remove_stop_words(query, additional_stop_words=(class_title,))

            print(f"Matching query: {query} with hints: {hints}")

            for npz_file in npz_files:
                # Two-pass over the same npz bytes: first pass runs CLIP matching
                # to get top patch indices/similarities, second pass loads the raw
                # npz arrays and merges them with match results into a single dict.
                # BytesIO is reopened each time because np.load/match consume the stream.
                with BytesIO(npz_file) as in_buffer:
                    results = self.multimodal.match(npz_file=in_buffer, query=query, top_k=3, hints=hints,
                                               show_plots=False)
                with BytesIO(npz_file) as in_buffer:
                    ret[key].append(dict(np.load(in_buffer, allow_pickle=True), **results))

        self.kp_initialized_empty = False
        return ret

    def backproject_kps(self, mesh: Any, fragments: Any, cameras: CamerasBase, T: Any, kps: Any) -> Pointclouds:
        """Transform matched patch centers from sampled space to mesh space via affine fit."""
        all_centers = []
        for kp in kps:
            mesh = mesh.to(device=cameras.device)
            pts = mesh.verts_packed()[kp['sample_to_input_idx']]
            points_sampled = torch.as_tensor(kp['points_sampled'], device=pts.device)
            A, t = find_affine_transform(points_sampled, pts)
            top_centers = torch.as_tensor(kp['top_centers'], device=pts.device)
            all_centers.append(top_centers @ A.T + t)
        return Pointclouds(points=all_centers, features=None)

    def aggregate_kps(self, mesh: Any, kps: Pointclouds, max_points: int = 10000, kp_prompt: str = '', last_kp_cache: Any = None) -> Pointclouds:
        """Pass through keypoints without clustering."""
        return kps

