"""PatchAlign3D reference-view baseline using few-shot patch feature matching."""

from typing import Any

from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes, Pointclouds
import torch
import numpy as np
from io import BytesIO

from zerokey.io.kpnet import RefIO
from patchalign3d.inference.explore_pc_patches import PatchExplorer
from zerokey.generators.kpnet import KPNetGenerator
from zerokey.generators.patchalign3d import PatchAlign3DGenerator


class PatchAlign3DRefIO(RefIO):
    """PatchAlign3D-specific reference I/O (inherits all behavior from RefIO)."""
    pass


class PatchAlign3DRefGenerator(PatchAlign3DGenerator, KPNetGenerator[PatchAlign3DRefIO, PatchExplorer]):
    """PatchAlign3D generator using reference-view features for keypoint matching.

    Extracts patch features from a single view and matches them against
    averaged reference features accumulated from ground-truth annotations.
    Uses diamond inheritance with PatchAlign3DGenerator for patch extraction
    and backprojection.
    """
    io: PatchAlign3DRefIO

    def process_kp_list(self, mesh: Meshes, fragments: Any, R: CamerasBase, T: Any, images: torch.Tensor, kp_list: dict[Any, Any], class_title: str, mesh_id: str, prompt_idx: int | slice = 0) -> dict[frozenset[Any], Pointclouds]:
        """Extract patches from a single view and match against reference features.

        First attempts to collect reference features via sample_reference_view.
        Once enough references are accumulated, matches each keypoint's reference
        feature against patch embeddings using cosine similarity.
        """
        with BytesIO() as in_buffer:
            np.savez(in_buffer, images[0].cpu().numpy())
            in_buffer.seek(0)
            npz_file = self.multimodal.extract(input=in_buffer, output=None)

        assert isinstance(npz_file, bytes)
        # sample_reference_view uses exception-based control flow:
        # yields False → caller raises LookupError → context manager accumulates
        # reference features and raises IOError to skip this mesh.
        # yields True → enough references collected, proceed to matching.
        with self.io.sample_reference_view(npz_file, kp_list, mesh, class_title) as used_as_reference:
            if not used_as_reference:
                raise LookupError("No reference view found")

            centers = []
            # Match query
            for semantic_id, _kp in kp_list.items():
                text_feat = self.io.get_reference_features(class_title, semantic_id).to(self.device)

                assert isinstance(npz_file, bytes)
                with BytesIO(npz_file) as in_buffer:
                    d = dict(np.load(in_buffer, allow_pickle=True))

                # Use projected features if available, otherwise fall back to raw embeddings
                patch_feat = d.get('patch_feat', d.get('patch_emb', None))
                if patch_feat is None:
                    raise ValueError("Neither patch_feat nor patch_emb found in npz file")

                # Compute similarities
                patch_feat_torch = torch.from_numpy(patch_feat).float().to(self.device)
                similarities = torch.einsum('gd,d->g', patch_feat_torch, text_feat)
                distances = 1 - similarities

                similarities_np = similarities.cpu().numpy()
                distances_np = distances.cpu().numpy()

                print(f"\n✓ Computed similarities for {len(patch_feat)} patches")
                print(f"  Similarity range: [{similarities_np.min():.4f}, {similarities_np.max():.4f}]")
                print(f"  Distance range:   [{distances_np.min():.4f}, {distances_np.max():.4f}]")

                # Find top matches
                top_indices = np.argsort(distances_np)[0]
                centers.append(d['patch_centers'][top_indices])

        d['top_centers'] = centers
        ret = {frozenset(kp_list.keys()): self.backproject_kps(mesh, fragments, R, T, (d,))}

        return ret
