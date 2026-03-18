"""PatchAlign3D + ZeroKey hybrid generator combining patch features with MLLM detection."""

from collections import defaultdict, UserDict
from dataclasses import dataclass
from typing import override, Any
import math

from pytorch3d.renderer import CamerasBase
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points
import torch
import numpy as np
from io import BytesIO
from sklearn.decomposition import PCA
from fast_hdbscan import HDBSCAN

from zerokey.rendering import camera_from_eye_at_up
from zerokey.models import Molmo
from patchalign3d.inference.explore_pc_patches import PatchExplorer
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.generators.patchalign3d import find_affine_transform, remove_stop_words
from zerokey.io.kpnet import KPNetIO


@dataclass(frozen=True)
class FragmentsWithPts(Fragments):
    """Extends Fragments with per-view projected vertices, cameras, and lights."""
    pts: torch.Tensor = torch.empty(0)
    cameras: CamerasBase | None = None
    lights: PointLights | None = None


class SuperDict(UserDict):
    """Dict subclass that carries npz patch data and per-query match results."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.npz_files: list[bytes] = []
        self.match_results: dict[str, list[dict]] = defaultdict(list)


class PatchAlign3DZeroKeyGenerator(KPNetGenerator[KPNetIO, Molmo]):
    """
    PatchAlign3D generator variant combining PatchAlign3D patch features with ZeroKey pipeline.

    This class inherits from KPNetGenerator and uses:
    - Molmo (via inheritance) for MLLM-based point detection
    - PatchExplorer (self.patch_explorer) for patch-based feature extraction
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, scale=1, **kwargs)
        self.vis: bool = False
        self.patch_explorer: PatchExplorer = PatchExplorer(cuda_device=str(self.device))

    @override
    def views_from_model(self, mesh: Any, views: Any, batch_size: int | None = None, device: str | torch.device = "cuda") -> tuple[torch.Tensor, 'FragmentsWithPts', CamerasBase, Any]:
        """Render multi-view images and project mesh vertices into each view's camera space.

        Extends the base views_from_model by:
          1. Calling super() to render images and get fragments/cameras/lights.
          2. Reconstructing per-view cameras from raw viewpoint positions, centered
             on the mesh centroid via camera_from_eye_at_up().
          3. Projecting all mesh vertices into each camera's view coordinate frame,
             producing pts of shape [num_views, num_verts, 3].
          4. Bundling everything into a FragmentsWithPts frozen dataclass that
             extends Fragments with pts, cameras, and lights fields.

        The pts tensor is later used by init_kps_batch to extract PatchAlign3D
        patch features in camera space for each view.

        Returns:
            images: Rendered views [B, C, H, W]
            fragments: FragmentsWithPts with zbuf, pts, cameras, lights
            cameras: FoVPerspectiveCameras
            lights: PointLights
        """
        images, fragments, cameras, lights = super().views_from_model(mesh, views, batch_size=batch_size, device=device)
        # images: Tensor [B, C, H, W] — rendered RGB images per view
        # fragments: Fragments with zbuf [B, H, W, K], pix_to_face [B, H, W, K], etc.
        # cameras: FoVPerspectiveCameras (batch of B cameras)
        # lights: PointLights

        # Reconstruct cameras from raw viewpoints.
        # Viewpoints are offsets from origin; shift them to be relative to mesh centroid.
        mesh = mesh.to(device=device)
        with torch.no_grad():
            target: torch.Tensor = mesh.verts_packed().mean(dim=0, keepdim=True)  # [1, 3]
            views = torch.as_tensor(views, dtype=target.dtype, device=target.device) + target  # [B, 3]
        # Build FoVPerspectiveCameras looking at the centroid from each viewpoint
        views = camera_from_eye_at_up(views, target, device=device)  # list of B FoVPerspectiveCameras

        # Transform mesh vertices into each camera's view coordinate frame.
        # mesh.verts_packed(): [V, 3] — all mesh vertices in world space
        # Result: [B, V, 3] — each view has all vertices in its local camera space.
        pts: torch.Tensor = torch.stack([view.get_world_to_view_transform().transform_points(mesh.verts_packed()) for view in views])

        # Bundle original fragments fields + pts/cameras/lights into extended dataclass
        fragments = FragmentsWithPts(
            pix_to_face=fragments.pix_to_face,
            zbuf=fragments.zbuf,
            bary_coords=fragments.bary_coords,
            dists=fragments.dists,
            pts=pts,
            cameras=cameras,
            lights=lights,
        )

        return images, fragments, cameras, lights

    @override
    def init_kps_batch(self, tensor_images: torch.Tensor, fragments: Any, kp: dict[Any, Any], class_title: str, mesh_id: str) -> 'SuperDict':
        """Pre-compute patch features and text-based matches for all views and prompts.

        Overrides the base init_kps_batch to:
          1. Extract PatchAlign3D patch features from the per-view projected vertices
             (fragments.pts) — each view's mesh vertices in camera space are saved as
             an npz and processed through PatchExplorer.extract().
          2. For each keypoint prompt, match it against the extracted patch features
             using CLIP text embeddings via PatchExplorer.match(). Results include
             top matching patch centers and their indices.
          3. Store everything in a SuperDict (npz_files + match_results) which is
             passed to aggregate_kps as last_kp_cache.

        Sets kp_initialized_empty = False so process_kp_list skips MLLM detection
        and goes straight to backproject_kps + aggregate_kps.
        """
        ret: SuperDict = SuperDict(super().init_kps_batch(tensor_images, fragments, kp, class_title, mesh_id))
        assert isinstance(fragments, FragmentsWithPts)

        # Step 1: Extract patch features for each view.
        # fragments.pts: [B, V, 3] — mesh vertices in each camera's coordinate frame.
        # Each image (one view's vertices): [V, 3] float32.
        for image in fragments.pts:
            with BytesIO() as in_buffer:
                # Serialize view-space vertices as npz for PatchExplorer input
                np.savez(in_buffer, image.cpu().numpy())  # npz with single array [V, 3]
                in_buffer.seek(0)
                # extract() returns bytes (npz) containing:
                #   patch_emb [G, C], patch_centers [G, 3], patch_indices [G, M],
                #   points_sampled [S, 3], sample_to_input_idx [S],
                #   patch_feat [G, D] (if CLIP projection available)
                npz_result = self.patch_explorer.extract(input=in_buffer, output=None)
                assert isinstance(npz_result, bytes)
                npz_bytes: bytes = npz_result
                ret.npz_files.append(npz_bytes)

        # Step 2: Match each keypoint prompt against extracted patch features.
        # kp: dict[semantic_id, tuple[str, ...]] — values are (query, *optional_hints) per semantic ID.
        # The query is matched via CLIP text embeddings against patch embeddings.
        for query, *hints in kp.values():
            key: str = query
            # Rotate hints: if extra strings provided, last becomes the query
            if hints:
                *hints, query = query, *hints
            if not hints:
                query, *hints = (query,)

            # Remove common stop words and class name for cleaner CLIP matching
            query = remove_stop_words(query, additional_stop_words=(class_title,))

            print(f"Matching query: {query} with hints: {hints}")

            # Match against each view's patch features
            for npz_file in ret.npz_files:
                with BytesIO(npz_file) as in_buffer:
                    # match() returns dict with keys:
                    #   'similarities' [G], 'distances' [G], 'top_indices' [K],
                    #   'top_patches' [K, M, 3], 'top_centers' [K, 3]
                    results: dict = self.patch_explorer.match(npz_file=in_buffer, query=query, top_k=np.iinfo(np.int_).max,
                                                             hints=hints, show_plots=False)
                with BytesIO(npz_file) as in_buffer:
                    # Merge raw npz data with match results into a single dict per view.
                    # Result dict has both npz fields (patch_feat, points_sampled, etc.)
                    # and match fields (top_indices, distances, etc.).
                    ret.match_results[key].append(dict(np.load(in_buffer, allow_pickle=True), **results))

        # Signal to process_kp_list that we've pre-computed results — skip MLLM detection
        self.kp_initialized_empty = False
        return ret

    @override
    def aggregate_kps(self, mesh: Any, pts_3d: Pointclouds, max_points: int = 10000, kp_prompt: str = '', last_kp_cache: Any = None) -> Pointclouds:
        """Aggregate keypoints using CLIP-enhanced HDBSCAN clustering.

        Combines two signals from PatchAlign3D patch matching to improve clustering:

        1. **Weight adjustment** via CLIP distance (nn_clip_dists):
           Each point's alpha weight is scaled by a CLIP-derived multiplier based on
           how well its source view's nearest patch matches the text query. Points near
           well-matched patches get boosted; poorly-matched points are suppressed.
           Uses percentile normalization + pow(2) sharpening with a 0.1x floor.

        2. **Augmented clustering dimensions** via CLIP features (nn_feats):
           Per-point CLIP features [M, B, D] are PCA-reduced to 3 components across
           all views, then averaged per point. The resulting [M, 3] semantic features
           are concatenated with 3D coordinates (scaled to match spatial spread) to
           form a 6D clustering input for HDBSCAN.

        The patch top_centers are in the sampled point cloud's coordinate space,
        so they are transformed to mesh space via an affine transform fitted between
        the sampled points and the original mesh vertices.

        Weighted centroids are computed in 3D space (not 6D) using adjusted weights,
        with the same multi-view support threshold as the base class.
        """
        assert isinstance(last_kp_cache, SuperDict)

        try:
            return self._aggregate_kps_clip(mesh, pts_3d, max_points, kp_prompt, last_kp_cache)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[WARN] CUDA OOM in CLIP-enhanced aggregate_kps for '{kp_prompt}', falling back to base method")
            return super().aggregate_kps(mesh, pts_3d, max_points=max_points, kp_prompt=kp_prompt, last_kp_cache=last_kp_cache)

    def _aggregate_kps_clip(self, mesh: Any, pts_3d: Pointclouds, max_points: int, kp_prompt: str, last_kp_cache: 'SuperDict') -> Pointclouds:
        """CLIP-enhanced HDBSCAN clustering (may raise torch.cuda.OutOfMemoryError)."""
        # Decode per-point features from the Pointclouds produced by backproject_kps.
        # Features are uint8 RGBA values packed as 4 channels:
        #   features[:, 0] = view index (uint8)
        #   features[:, 1] = point class ID within view (uint8, color-mapped)
        #   features[:, 2] = alpha/confidence weight (uint8)
        #   features[:, 3] = valid mask (uint8, 1 = ray faces camera)
        features = pts_3d.features_packed()  # [N, 4], dtype=uint8
        assert features is not None
        valid_mask: torch.Tensor = features[:, 3].bool()           # [N] bool
        view_indices: torch.Tensor = features[valid_mask, 0].long()  # [M] int64, view index per valid point

        # Cross-reference with patch match results for this prompt.
        # last_kp_cache.match_results[kp_prompt] is a list of dicts, one per view,
        # each containing merged npz data + match() output. Key fields per view dict:
        #   'points_sampled':       ndarray [S, 3] float32    — S sampled points in normalized space
        #   'sample_to_input_idx':  ndarray [S]    int64      — maps sampled index -> original mesh vertex index
        #   'patch_feat':           ndarray [G, D] float32    — CLIP-projected features per patch (G patches, D dims)
        #   'patch_centers':        ndarray [G, 3] float32    — centroid of each patch in sampled space
        #   'top_indices':          ndarray [K]    int64      — indices of top-K matched patches (sorted by distance)
        #   'top_centers':          ndarray [K, 3] float32    — centroids of top-K patches (= patch_centers[top_indices])
        #   'distances':            ndarray [G]    float32    — 1 - cosine_similarity per patch to text query
        #   'similarities':         ndarray [G]    float32    — cosine similarity per patch to text query
        mesh = mesh.to(device=self.device)
        all_centers: list[torch.Tensor] = []     # each [K_v, 3], transformed top patch centers per view
        all_distances: list[torch.Tensor] = []   # each [K_v], CLIP distances for top patches per view
        all_features: list[torch.Tensor] = []    # each [K_v, D], CLIP features for top patches per view
        for _view_idx, kp in enumerate(last_kp_cache.match_results[kp_prompt]):
            # Transform patch top_centers from sampled space to mesh space.
            # find_affine_transform solves b = a @ A.T + t via least squares.
            mesh_pts: torch.Tensor = mesh.verts_packed()[kp['sample_to_input_idx']]  # [S, 3]
            points_sampled: torch.Tensor = torch.as_tensor(kp['points_sampled'], device=self.device)  # [S, 3]
            A, t = find_affine_transform(points_sampled, mesh_pts)  # A: [3, 3], t: [3]
            top_centers: torch.Tensor = torch.as_tensor(kp['top_centers'], device=self.device)  # [K, 3]
            all_centers.append(top_centers @ A.T + t)  # [K, 3] in mesh space

            top_indices: torch.Tensor = torch.as_tensor(kp['top_indices'], device=self.device)  # [K] int64
            distances: torch.Tensor = torch.as_tensor(kp['distances'], device=self.device)[top_indices]  # [K]
            all_distances.append(distances)
            top_feat: torch.Tensor = torch.as_tensor(kp['patch_feat'], device=self.device)[top_indices]  # [K, D]
            all_features.append(top_feat)

        # Stack per-view results into tensors with a view dimension.
        # B = num_views, G = num_patches per view (same across views).
        num_views: int = len(all_centers)
        centers: torch.Tensor = torch.stack(all_centers, dim=0)    # [B, G, 3], patch centers in mesh space
        dists: torch.Tensor = torch.stack(all_distances, dim=0)    # [B, G], CLIP distance per patch
        feats: torch.Tensor = torch.stack(all_features, dim=0)     # [B, G, D], CLIP feature per patch

        # For each valid backprojected point, find its closest patch center per view.
        # Broadcast points [B, M, 3] against each view's centers [B, G, 3].
        points: torch.Tensor = pts_3d.points_packed()[valid_mask]   # [M, 3]
        M: int = points.size(0)
        if M <= 1:
            return Pointclouds(points[None])

        points_batch: torch.Tensor = points[None].expand(num_views, -1, -1)  # [B, M, 3]
        knn = knn_points(points_batch, centers, K=1)                # dists: [B, M, 1], idx: [B, M, 1]
        nn_idx: torch.Tensor = knn.idx[:, :, 0]                    # [B, M], index into G patches per view
        # Gather CLIP distances and features for each point's nearest patch per view
        nn_clip_dists: torch.Tensor = dists.gather(1, nn_idx)       # [B, M], CLIP distance per view
        nn_feats: torch.Tensor = feats.gather(1, nn_idx.unsqueeze(-1).expand(-1, -1, feats.size(-1)))  # [B, M, D]
        # Transpose to point-major layout: each point has B view-wise results
        nn_clip_dists = nn_clip_dists.permute(1, 0)                 # [M, B], CLIP distance per view
        nn_feats = nn_feats.permute(1, 0, 2)                       # [M, B, D], CLIP feature per view

        # --- Step 1: Adjust alpha weights using own-view CLIP distance ---
        # Each backprojected point came from one view; use that view's CLIP distance.
        point_idx: torch.Tensor = torch.arange(M, device=self.device)
        own_clip_dist: torch.Tensor = nn_clip_dists[point_idx, view_indices.clamp(max=num_views - 1)]  # [M]

        # Percentile normalization: lower CLIP distance = better match = higher score
        clip_min: torch.Tensor = own_clip_dist.min()
        clip_max: torch.Tensor = own_clip_dist.max()
        if clip_max > clip_min:
            clip_score: torch.Tensor = 1.0 - (own_clip_dist - clip_min) / (clip_max - clip_min)  # [M], range [0, 1]
        else:
            clip_score = torch.ones(M, device=self.device)

        # Sharpen: amplify gap between good and bad matches; floor at 0.1x
        clip_score = clip_score.pow(2)
        weight_multiplier: torch.Tensor = 0.1 + 0.9 * clip_score   # [M], range [0.1, 1.0]

        # Scale original alpha weights by CLIP-based multiplier
        raw_weights: torch.Tensor = features[valid_mask, 2].float()  # [M], original uint8 alpha
        adjusted_weights: torch.Tensor = (raw_weights * weight_multiplier).clamp(1, 255)  # [M]

        # --- Step 2: PCA-reduce CLIP features and mean across views ---
        # Flatten [M, B, D] -> [M*B, D], fit PCA to 3 components, reshape back, then average.
        feat_D: int = nn_feats.size(-1)
        feats_flat: np.ndarray = nn_feats.reshape(M * num_views, feat_D).cpu().numpy()  # [M*B, D]
        pca = PCA(n_components=3)
        feats_reduced: np.ndarray = pca.fit_transform(feats_flat)                        # [M*B, 3]
        feats_reduced_t: torch.Tensor = torch.from_numpy(feats_reduced).to(device=self.device, dtype=points.dtype)
        feats_mean: torch.Tensor = feats_reduced_t.reshape(M, num_views, 3).mean(dim=1)  # [M, 3]

        # --- Step 3: Concatenate scaled PCA features with 3D coordinates ---
        feat_std: float = float(feats_mean.std())
        feat_scale: float = float(points.std()) / feat_std if feat_std > 0 else 1.0
        clustering_input: torch.Tensor = torch.cat([points, feats_mean * feat_scale], dim=-1)  # [M, 6]

        # --- Step 4: Run HDBSCAN on 6D space with adjusted weights ---
        per_point_mean_weight: torch.Tensor = features[:, 2].mean(dtype=torch.float32)
        np_weights: np.ndarray = (adjusted_weights / per_point_mean_weight).cpu().numpy()  # [M]
        np_input: np.ndarray = clustering_input.cpu().numpy()                               # [M, 6]

        min_cluster_size: int = 10
        clustered_masks: list[np.ndarray] = []
        while not clustered_masks and min_cluster_size >= 2:
            clustering = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=True)
            cluster_assign: np.ndarray = clustering.fit_predict(np_input, sample_weight=np_weights)
            clustered_masks = [cluster_assign == i for i in np.unique(cluster_assign) if i != -1]
            min_cluster_size = math.ceil(min_cluster_size / 2)

        # Fallback: if no clusters found, treat all points as one cluster
        if not clustered_masks:
            clustered_masks = [np.ones(M, dtype=bool)]

        # --- Step 5: Compute weighted centroids in 3D space only ---
        # Use original 3D positions + CLIP-adjusted weights for centroid computation.
        clustered_rawpts = Pointclouds(
            points=[points[mask] for mask in clustered_masks],
            features=[adjusted_weights[mask].unsqueeze(-1) for mask in clustered_masks])

        # Multi-view support threshold: cluster weight must exceed 2x max single-class weight.
        # Reinterpret first 2 uint8 channels as int16 for packed (view, class) identifier.
        features_id: torch.Tensor = features[valid_mask, :2].contiguous().view(torch.int16).view(-1)  # [M]
        per_point_max_weight: torch.Tensor = max(
            adjusted_weights[features_id == m].sum() for m in features_id.unique())

        # Helper to get features with type narrowing (features_packed() is never None for valid Pointclouds)
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
