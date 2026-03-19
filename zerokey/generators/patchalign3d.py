"""PatchAlign3D generator: unified patch-level 3D keypoint detection with three operating modes.

Modes:
  - patch:    Pure patch-based (PatchExplorer extract+match, affine backprojection, no clustering)
  - zerokey:  Hybrid MLLM+patch (Molmo detection + PatchExplorer features + CLIP-enhanced HDBSCAN)
  - ref:      Reference-view few-shot matching (accumulate reference features, cosine similarity)
"""

from collections import defaultdict, UserDict
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from io import BytesIO
from typing import Any, Iterable, override
import math

from pytorch3d.ops import knn_points
from pytorch3d.renderer import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes, Pointclouds
import torch
import numpy as np
from fast_hdbscan import HDBSCAN
from sklearn.decomposition import PCA

from zerokey.rendering import camera_from_eye_at_up
from patchalign3d.inference.explore_pc_patches import PatchExplorer
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.io.kpnet import KPNetIO, RefIO


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def find_affine_transform(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find optimal affine transformation from a to b.  Solves: b = a @ A.T + t"""
    N = a.shape[0]
    a_homo = torch.cat([a, torch.ones(N, 1, device=a.device, dtype=a.dtype)], dim=1)
    M = torch.linalg.lstsq(a_homo, b).solution
    return M[:3, :].T, M[3, :]


def remove_stop_words(query: str, additional_stop_words: tuple[str, ...] = ()) -> str:
    """Remove common stop words from a query string."""
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
    return ' '.join(word for word in query.split() if word.lower() not in stop_words)


def transform_to_mesh_space(kp: Any, mesh: Meshes, device: torch.device,
                            centers_key: str = 'top_centers') -> torch.Tensor:
    """Transform points from PatchExplorer's sampled space to mesh space via affine fit.

    Args:
        kp: Dict-like with 'sample_to_input_idx', 'points_sampled', and *centers_key*.
        mesh: Meshes whose verts_packed provides the target coordinate frame.
        device: Torch device for tensor operations.
        centers_key: Key into *kp* for the centers to transform (default 'top_centers').

    Returns:
        Transformed centers [K, 3] in mesh-space coordinates.
    """
    verts = mesh.verts_packed()
    assert verts is not None
    mesh_pts = verts[kp['sample_to_input_idx']]
    points_sampled = torch.as_tensor(kp['points_sampled'], device=device)
    A, t = find_affine_transform(points_sampled, mesh_pts)
    centers = torch.as_tensor(kp[centers_key], device=device)
    return centers @ A.T + t


def match_reference_features(
    io: RefIO, npz_file: bytes, kp_list: dict[Any, Any],
    class_title: str, device: torch.device,
) -> dict[str, Any]:
    """Load npz patch data and find best-matching patch per semantic keypoint.

    For each semantic ID in *kp_list*, retrieves the accumulated reference
    feature from *io* and matches it against patch embeddings via cosine
    similarity.  Shared by PatchAlign3D ref mode and ULIP2RefGenerator.

    Returns:
        The npz data dict with an added 'top_centers' key containing the
        best-matching patch center per semantic keypoint.
    """
    with BytesIO(npz_file) as in_buffer:
        d = dict(np.load(in_buffer, allow_pickle=True))

    patch_feat = d.get('patch_feat', d.get('patch_emb', None))
    if patch_feat is None:
        raise ValueError("Neither patch_feat nor patch_emb found in npz file")
    patch_feat_torch = torch.from_numpy(patch_feat).float().to(device)

    centers = []
    for semantic_id, _kp in kp_list.items():
        text_feat = io.get_reference_features(class_title, semantic_id).to(device)
        distances_np = (1 - torch.einsum('gd,d->g', patch_feat_torch, text_feat)).cpu().numpy()

        print(f"\n✓ Computed similarities for {len(patch_feat)} patches")
        print(f"  Similarity range: [{(1 - distances_np).min():.4f}, {(1 - distances_np).max():.4f}]")
        print(f"  Distance range:   [{distances_np.min():.4f}, {distances_np.max():.4f}]")

        centers.append(d['patch_centers'][np.argsort(distances_np)[0]])

    d['top_centers'] = centers
    return d


# ---------------------------------------------------------------------------
# Mode enum and helper types
# ---------------------------------------------------------------------------

class PatchAlign3DMode(StrEnum):
    """Operating mode for PatchAlign3DGenerator."""
    PATCH = 'patch'
    ZEROKEY = 'zerokey'
    REF = 'ref'


@dataclass(frozen=True)
class FragmentsWithPts(Fragments):
    """Extends Fragments with per-view projected vertices."""
    pts: torch.Tensor = torch.empty(0)


class SuperDict(UserDict):
    """Dict subclass that carries npz patch data and per-query match results."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.npz_files: list[bytes] = []
        self.match_results: dict[str, list[dict]] = defaultdict(list)


class PatchAlign3DRefIO(RefIO):
    """PatchAlign3D-specific reference I/O (inherits all behavior from RefIO)."""
    pass


# ---------------------------------------------------------------------------
# Unified generator
# ---------------------------------------------------------------------------

class PatchAlign3DGenerator(KPNetGenerator[KPNetIO, PatchExplorer]):
    """Unified PatchAlign3D generator with mode-dispatched behavior.

    Modes:
        PATCH:   Pure patch-based — extract patch features, CLIP text match, affine
                 backprojection, no clustering.
        ZEROKEY: Hybrid — render images via Molmo MLLM for 2D detection, plus
                 PatchExplorer patch features for CLIP-enhanced HDBSCAN clustering.
        REF:     Reference-view few-shot — accumulate patch features from reference
                 views, match via cosine similarity.
    """

    def __init__(self, *args: Any, mode: PatchAlign3DMode = PatchAlign3DMode.PATCH, **kwargs: Any) -> None:
        self.mode = PatchAlign3DMode(mode)
        super().__init__(*args, scale=1, **kwargs)
        self.vis = False

    # -- Lazy model/IO properties, dispatched by mode --

    @cached_property
    def multimodal(self) -> PatchExplorer:
        """PatchExplorer instance; also initializes Molmo VLM in zerokey mode."""
        return PatchExplorer(cuda_device=str(self.device),
                             molmo=self.mode == PatchAlign3DMode.ZEROKEY)

    @cached_property
    def io(self) -> Any:
        """I/O handler: PatchAlign3DRefIO for ref mode, standard KPNetIO otherwise."""
        if self.mode == PatchAlign3DMode.REF:
            return PatchAlign3DRefIO(self.log_dir / self.expname)
        return self.KPIO(self.log_dir / self.expname)

    # -- Shared helpers --

    @staticmethod
    def _project_verts(mesh: Meshes, views: Any,
                       device: str | torch.device = "cuda") -> tuple[torch.Tensor, Any]:
        """Project all mesh vertices into each camera's view coordinate frame.

        Returns:
            pts: [B, V, 3] per-view vertices in camera space
            cameras: FoVPerspectiveCameras
        """
        mesh = mesh.to(device=device)
        with torch.no_grad():
            verts = mesh.verts_packed()
            assert verts is not None
            target = verts.mean(dim=0, keepdim=True)
            views = torch.as_tensor(views, dtype=target.dtype, device=target.device) + target
        cameras = camera_from_eye_at_up(views, target, device=device)
        pts = torch.stack([
            cam.get_world_to_view_transform().transform_points(mesh.verts_packed())
            for cam in cameras
        ])
        return pts, cameras

    def _extract_and_match_patches(
        self,
        pts_per_view: Iterable[torch.Tensor],
        kp: dict[Any, Any],
        class_title: str,
        *,
        top_k: int = 3,
    ) -> tuple[list[bytes], defaultdict[str, list[dict[str, Any]]]]:
        """Extract patch features per view and match each keypoint query.

        Returns:
            (npz_files, match_results) where npz_files is raw bytes per view and
            match_results maps query string -> list of dicts (one per view).
        """
        npz_files: list[bytes] = []
        for image in pts_per_view:
            with BytesIO() as in_buffer:
                np.savez(in_buffer, image.cpu().numpy())
                in_buffer.seek(0)
                result = self.multimodal.extract(input=in_buffer, output=None)
                assert isinstance(result, bytes)
                npz_files.append(result)

        match_results: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for query, *hints in kp.values():
            key = query
            if hints:
                *hints, query = query, *hints
            if not hints:
                query, *hints = (query,)

            query = remove_stop_words(query, additional_stop_words=(class_title,))
            print(f"Matching query: {query} with hints: {hints}")

            for npz_file in npz_files:
                with BytesIO(npz_file) as in_buffer:
                    results = self.multimodal.match(
                        npz_file=in_buffer, query=query, top_k=top_k,
                        hints=hints, show_plots=False)
                with BytesIO(npz_file) as in_buffer:
                    match_results[key].append(dict(np.load(in_buffer, allow_pickle=True), **results))

        return npz_files, match_results

    # -- Pipeline overrides --

    @override
    def sample_view_points(self, dist: float, partition: int | None = None) -> Any:
        return super().sample_view_points(dist, partition=-1)

    @override
    def views_from_model(self, mesh: Any, views: Any, batch_size: int | None = None,
                         device: str | torch.device = "cuda") -> tuple:
        if self.mode == PatchAlign3DMode.ZEROKEY:
            # Render images via base class, then augment fragments with projected vertices
            images, fragments, cameras, lights = KPNetGenerator.views_from_model(
                self, mesh, views, batch_size=batch_size, device=device)
            pts, _cam_views = self._project_verts(mesh, views, device=device)
            fragments = FragmentsWithPts(
                pix_to_face=fragments.pix_to_face, zbuf=fragments.zbuf,
                bary_coords=fragments.bary_coords, dists=fragments.dists,
                pts=pts)
            return images, fragments, cameras, lights

        # patch/ref: vertex projection only, no rendering
        pts, cameras = self._project_verts(mesh, views, device=device)
        return pts, None, cameras, None

    @override
    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any,
                       kp: dict[Any, Any], class_title: str,
                       mesh_id: str) -> Any:
        if self.mode == PatchAlign3DMode.PATCH:
            _npz_files, match_results = self._extract_and_match_patches(
                tensor_images, kp, class_title, top_k=3)
            self.kp_initialized_empty = False
            return match_results

        if self.mode == PatchAlign3DMode.ZEROKEY:
            ret = SuperDict(super().init_kps_batch(tensor_images, fragments, kp, class_title, mesh_id))
            assert isinstance(fragments, FragmentsWithPts)
            npz_files, match_results = self._extract_and_match_patches(
                fragments.pts, kp, class_title, top_k=np.iinfo(np.int_).max)
            ret.npz_files = npz_files
            ret.match_results = dict(match_results)
            self.kp_initialized_empty = False
            return ret

        # ref mode: base behavior (process_kp_list is overridden)
        return super().init_kps_batch(tensor_images, fragments, kp, class_title, mesh_id)

    @override
    def backproject_kps(self, mesh: Any, fragments: Any, cameras: CamerasBase,
                        T: Any, kps: Any) -> Pointclouds:
        if self.mode == PatchAlign3DMode.ZEROKEY:
            return KPNetGenerator.backproject_kps(self, mesh, fragments, cameras, T, kps)
        # patch/ref: affine transform from sampled space to mesh space
        mesh = mesh.to(device=cameras.device)
        return Pointclouds(
            points=[transform_to_mesh_space(kp, mesh, cameras.device) for kp in kps],
            features=None)

    @override
    def aggregate_kps(self, mesh: Any, kps: Pointclouds, max_points: int = 10000,
                      kp_prompt: str = '', last_kp_cache: Any = None) -> Pointclouds:
        if self.mode != PatchAlign3DMode.ZEROKEY:
            return kps  # patch mode: pass-through

        assert isinstance(last_kp_cache, SuperDict)
        try:
            return self._aggregate_clip(mesh, kps, kp_prompt, last_kp_cache)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[WARN] CUDA OOM in CLIP-enhanced aggregate_kps for '{kp_prompt}', falling back to base")
            return KPNetGenerator.aggregate_kps(
                self, mesh, kps, max_points=max_points,
                kp_prompt=kp_prompt, last_kp_cache=last_kp_cache)

    def _aggregate_clip(self, mesh: Any, pts_3d: Pointclouds,
                        kp_prompt: str, last_kp_cache: SuperDict) -> Pointclouds:
        """CLIP-enhanced HDBSCAN clustering.

        Combines two signals:
        1. Weight adjustment via CLIP distance: points near well-matched patches get boosted.
        2. Augmented 6D clustering: PCA-reduced CLIP features + 3D coordinates.
        """
        features = pts_3d.features_packed()
        assert features is not None
        valid_mask: torch.Tensor = features[:, 3].bool()
        view_indices: torch.Tensor = features[valid_mask, 0].long()

        # Transform patch centers and gather CLIP distances/features per view
        mesh = mesh.to(device=self.device)
        all_centers: list[torch.Tensor] = []
        all_distances: list[torch.Tensor] = []
        all_features: list[torch.Tensor] = []
        for kp in last_kp_cache.match_results[kp_prompt]:
            all_centers.append(transform_to_mesh_space(kp, mesh, self.device))
            top_indices = torch.as_tensor(kp['top_indices'], device=self.device)
            all_distances.append(torch.as_tensor(kp['distances'], device=self.device)[top_indices])
            all_features.append(torch.as_tensor(kp['patch_feat'], device=self.device)[top_indices])

        num_views = len(all_centers)
        centers = torch.stack(all_centers)       # [B, G, 3]
        dists = torch.stack(all_distances)       # [B, G]
        feats = torch.stack(all_features)        # [B, G, D]

        points: torch.Tensor = pts_3d.points_packed()[valid_mask]  # [M, 3]
        M = points.size(0)
        if M <= 1:
            return Pointclouds(points[None])

        # For each point, find nearest patch center per view
        knn = knn_points(points[None].expand(num_views, -1, -1), centers, K=1)
        nn_idx = knn.idx[:, :, 0]                                  # [B, M]
        nn_clip_dists = dists.gather(1, nn_idx).permute(1, 0)      # [M, B]
        nn_feats = feats.gather(1, nn_idx.unsqueeze(-1).expand(-1, -1, feats.size(-1))).permute(1, 0, 2)  # [M, B, D]

        # Step 1: Adjust alpha weights using own-view CLIP distance
        own_clip_dist = nn_clip_dists[torch.arange(M, device=self.device), view_indices.clamp(max=num_views - 1)]
        clip_min, clip_max = own_clip_dist.min(), own_clip_dist.max()
        clip_score = (1.0 - (own_clip_dist - clip_min) / (clip_max - clip_min)).pow(2) if clip_max > clip_min else torch.ones(M, device=self.device)
        adjusted_weights = (features[valid_mask, 2].float() * (0.1 + 0.9 * clip_score)).clamp(1, 255)

        # Step 2: PCA-reduce CLIP features and mean across views
        feats_flat = nn_feats.reshape(M * num_views, -1).cpu().numpy()
        feats_reduced = torch.from_numpy(PCA(n_components=3).fit_transform(feats_flat)).to(device=self.device, dtype=points.dtype)
        feats_mean = feats_reduced.reshape(M, num_views, 3).mean(dim=1)  # [M, 3]

        # Step 3: 6D clustering input = [xyz, scaled_pca_features]
        feat_std = float(feats_mean.std())
        feat_scale = float(points.std()) / feat_std if feat_std > 0 else 1.0
        np_input = torch.cat([points, feats_mean * feat_scale], dim=-1).cpu().numpy()
        np_weights = (adjusted_weights / features[:, 2].mean(dtype=torch.float32)).cpu().numpy()

        # Step 4: HDBSCAN with retry
        min_cluster_size = 10
        clustered_masks: list[np.ndarray] = []
        while not clustered_masks and min_cluster_size >= 2:
            cluster_assign = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True).fit_predict(np_input, sample_weight=np_weights)
            clustered_masks = [cluster_assign == i for i in np.unique(cluster_assign) if i != -1]
            min_cluster_size = math.ceil(min_cluster_size / 2)
        if not clustered_masks:
            clustered_masks = [np.ones(M, dtype=bool)]

        # Step 5: Weighted centroids with multi-view support threshold
        clustered_rawpts = Pointclouds(
            points=[points[mask] for mask in clustered_masks],
            features=[adjusted_weights[mask].unsqueeze(-1) for mask in clustered_masks])

        features_id = features[valid_mask, :2].contiguous().view(torch.int16).view(-1)
        per_point_max_weight: torch.Tensor = max(
            adjusted_weights[features_id == m].sum() for m in features_id.unique())

        def get_feats(p: Pointclouds) -> torch.Tensor:
            f = p.features_packed()
            assert f is not None
            return f

        clustered_pts = Pointclouds([
            (p.points_packed() * get_feats(p)).sum(dim=0, keepdim=True) / total_w
            for p in clustered_rawpts
            if (total_w := get_feats(p).sum()) > 2 * per_point_max_weight
        ])
        if not clustered_pts:
            clustered_pts = Pointclouds([
                (p.points_packed() * get_feats(p)).sum(dim=0, keepdim=True) / total_w
                for p in clustered_rawpts
                if (total_w := get_feats(p).sum())
            ])
        return clustered_pts

    # -- Reference-view mode --

    @override
    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any,
                        images: torch.Tensor, kp_list: dict[Any, Any], class_title: str,
                        mesh_id: str, prompt_idx: int | slice = 0) -> dict[frozenset[Any], Pointclouds]:
        if self.mode != PatchAlign3DMode.REF:
            return super().process_kp_list(mesh, fragments, R, T, images, kp_list, class_title, mesh_id, prompt_idx)

        assert isinstance(self.io, RefIO)
        with BytesIO() as in_buffer:
            np.savez(in_buffer, images[0].cpu().numpy())
            in_buffer.seek(0)
            npz_file = self.multimodal.extract(input=in_buffer, output=None)

        assert isinstance(npz_file, bytes)
        with self.io.sample_reference_view(npz_file, kp_list, mesh, class_title) as used_as_reference:
            if not used_as_reference:
                raise LookupError("No reference view found")
            d = match_reference_features(self.io, npz_file, kp_list, class_title, self.device)

        mesh = mesh.to(device=self.device)
        return {frozenset(kp_list.keys()): Pointclouds(
            points=[transform_to_mesh_space(d, mesh, self.device)],
            features=None)}
