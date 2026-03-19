#!/usr/bin/env python3
"""Explore Point Cloud Patches.

Extract and analyse point cloud patches from 3D models using Point-BERT,
then match them to natural-language queries via CLIP text embeddings.

Usage::

    python explore_pc_patches.py extract --input model.ply --ckpt checkpoint.pt
    python explore_pc_patches.py match --npz_file patches.npz --query "nose"
    python explore_pc_patches.py analyze --npz_file patches.npz
    python explore_pc_patches.py pipeline --input model.ply --query "nose"
"""

from __future__ import annotations

from io import BytesIO, IOBase
from typing import Any, BinaryIO, Dict, Optional, Sequence, Tuple, Union

from zerokey._defaults import POINTBERT_CKPT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import sys
from pathlib import Path
import torch
import fire
from zerokey.models.molmo import Molmo


class PatchExplorer(Molmo):
    """Explore point cloud patches with text-based matching.

    Inherits from Molmo so that zerokey mode can use both PatchExplorer
    feature extraction and Molmo MLLM detection through a single object.
    Pass ``molmo=True`` to also initialize the Molmo VLM.
    """

    def __init__(
        self,
        clip_model: str = 'ViT-bigG-14',
        clip_pretrained: str = 'laion2b_s39b_b160k',
        cuda_device: str | int = '0',
        molmo: bool = False,
    ) -> None:
        """Initialise the explorer and load the CLIP text encoder.

        Args:
            clip_model: OpenCLIP model architecture name.
            clip_pretrained: Pretrained weights tag for *clip_model*.
            cuda_device: CUDA device identifier (e.g. ``'0'`` or ``'cuda:0'``).
            molmo: If True, also initialize the Molmo VLM for keypoint detection.
        """
        if molmo:
            Molmo.__init__(self)
        # Normalise '0' -> 'cuda:0'; torch.device('0') is invalid
        if isinstance(cuda_device, str) and cuda_device.isdigit():
            cuda_device = f'cuda:{cuda_device}'
        self.device = torch.device(cuda_device)

        # Setup paths
        self._setup_paths()

        import open_clip  # type: ignore[import-untyped]
        from tools.eval_cli import LRUTextCache  # type: ignore[import-untyped]
        print(f"\nLoading CLIP: {clip_model} ({clip_pretrained})")
        clip_model_obj, _, _ = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained, device=self.device
        )
        clip_model_obj.eval()
        tokenizer = open_clip.get_tokenizer(clip_model)

        clip_dim: int = clip_model_obj.text_projection.shape[1]  # type: ignore[union-attr]
        print(f"CLIP loaded with {clip_dim}D text features")

        # Create text cache
        self.text_cache = LRUTextCache(
            device=self.device,
            backend='clip',
            clip_model=clip_model_obj,
            tokenizer=tokenizer,
            text_dim=int(clip_dim)
        )

        print(f"Device: {self.device}")

    def _setup_paths(self) -> None:
        """Add the ``patchalign3d/`` root to ``sys.path`` so that ``tools.*`` imports work."""
        script_dir = Path(__file__).parent
        seg_dir = script_dir.parent
        sys.path.insert(0, str(seg_dir))

    def extract(
        self,
        input: Union[str, os.PathLike[str], IOBase],
        ckpt: Union[str, os.PathLike] = POINTBERT_CKPT,
        output: Optional[Union[str, os.PathLike]] = None,
        arch: str = 'pointtransformer',
        num_group: int = 128,
        group_size: int = 32,
        npoints: int = 2048,
        no_swap_yz: bool = False,
        overwrite: bool = True,
        seed: int = 0
    ) -> bytes | str:
        """Extract patch features from a 3D model.

        Args:
            input: Path to input file (.ply, .glb, etc.)
            ckpt: Path to checkpoint file
            output: Output .npz file path (default: temp file)
            arch: Architecture name (default: 'pointtransformer')
            num_group: Number of patches (default: 128)
            group_size: Points per patch (default: 32)
            npoints: Number of sampled points (default: 2048)
            no_swap_yz: Don't swap Y/Z coordinates (default: False)
            overwrite: Overwrite existing output (default: True)
            seed: Random seed (default: 0)

        Returns:
            Path to output .npz file
        """
        from inference.extract_patch_features import main as extract_patch_features_main

        # Default output path
        if output is None:
            out_target: Union[BytesIO, Path] = BytesIO()
            fgetvalue = BytesIO.getvalue
        else:
            out_target = Path(output)
            fgetvalue = os.fspath

        print("=" * 80)
        print("EXTRACTING PATCH FEATURES")
        print("=" * 80)
        print(f"Input:      {input}")
        print(f"Checkpoint: {ckpt}")
        print(f"Output:     {out_target}")
        print(f"Device:     {self.device}")
        print(f"Patches:    {num_group} x {group_size} points")
        print(f"Sampled:    {npoints} points")

        # Extract features
        extract_patch_features_main(
            input=input,  # type: ignore[arg-type]
            ckpt=os.fspath(ckpt),
            output=out_target,
            arch=arch,
            num_group=num_group,
            group_size=group_size,
            npoints=npoints,
            device=str(self.device),
            no_swap_yz=no_swap_yz,
            seed=seed,
            overwrite=overwrite,
            save_raw_only=False
        )

        print(f"\n✓ Successfully extracted patches to {str(out_target)}")
        return fgetvalue(out_target)  # type: ignore[arg-type]

    def analyze(self, npz_file: str, show_plots: bool = True) -> Dict[str, Any]:
        """Analyse patch data from an ``.npz`` file.

        Args:
            npz_file: Path to ``.npz`` file containing patch data.
            show_plots: Whether to show matplotlib plots.

        Returns:
            Dictionary with keys ``patches``, ``patch_centers``,
            ``patch_emb``, ``patch_feat``, ``points_sampled``,
            ``coverage``, and ``patch_sizes``.
        """
        npz_path = Path(npz_file)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        print("=" * 80)
        print("ANALYZING PATCH DATA")
        print("=" * 80)
        print(f"File: {npz_file}")

        # Load data
        npz_data = np.load(str(npz_path), allow_pickle=True)

        # Extract arrays
        patch_emb = npz_data['patch_emb']
        patch_centers = npz_data['patch_centers']
        patch_indices = npz_data['patch_indices']
        points_sampled = npz_data['points_sampled']

        patch_feat = npz_data.get('patch_feat', None)

        print(f"\nData loaded:")
        print(f"  • patch_emb:     {patch_emb.shape}")
        if patch_feat is not None:
            print(f"  • patch_feat:    {patch_feat.shape}")
        print(f"  • patch_centers: {patch_centers.shape}")
        print(f"  • patch_indices: {patch_indices.shape}")
        print(f"  • points_sampled: {points_sampled.shape}")

        # Reconstruct patches
        num_patches = patch_indices.shape[0]
        points_per_patch = patch_indices.shape[1]
        patches = np.zeros((num_patches, points_per_patch, 3), dtype=np.float32)

        for i in range(num_patches):
            patches[i] = points_sampled[patch_indices[i]]

        # Statistics
        unique_indices = np.unique(patch_indices.flatten())
        coverage = len(unique_indices) / len(points_sampled) * 100

        patch_emb_norms = np.linalg.norm(patch_emb, axis=1)
        if patch_feat is not None:
            patch_feat_norms = np.linalg.norm(patch_feat, axis=1)

        # Calculate patch sizes
        patch_sizes = np.sqrt(
            ((patches - patch_centers[:, np.newaxis, :])**2).sum(axis=2)
        ).max(axis=1)

        print(f"\nStatistics:")
        print(f"  • Patch coverage: {len(unique_indices)}/{len(points_sampled)} ({coverage:.1f}%)")
        print(f"  • Point-BERT norm - Mean: {patch_emb_norms.mean():.4f}, Std: {patch_emb_norms.std():.4f}")
        if patch_feat is not None:
            print(f"  • Text feat norm  - Mean: {patch_feat_norms.mean():.4f}, Std: {patch_feat_norms.std():.4f}")
        print(f"  • Patch radius    - Mean: {patch_sizes.mean():.4f}, Std: {patch_sizes.std():.4f}")

        if show_plots:
            self._plot_analysis(points_sampled, patch_centers, patches, patch_emb, patch_feat)

        return {
            'patches': patches,
            'patch_centers': patch_centers,
            'patch_emb': patch_emb,
            'patch_feat': patch_feat,
            'points_sampled': points_sampled,
            'coverage': coverage,
            'patch_sizes': patch_sizes
        }

    def match(
        self,
        npz_file: str | bytes | os.PathLike[str] | BinaryIO | IOBase,
        query: str,
        top_k: int = 10,
        show_plots: bool = True,
        hints: Sequence[str] = ()
    ) -> Dict[str, Any]:
        """Match patches to a text query using CLIP cosine similarity.

        Args:
            npz_file: Path to ``.npz`` file containing patch data.
            query: Text query (e.g. ``"nose"``, ``"wheel"``).
            top_k: Number of top matches to return.
            show_plots: Whether to show matplotlib plots.
            hints: Extra text strings appended to the prompt template.

        Returns:
            Dictionary with ``similarities``, ``distances``,
            ``top_indices``, ``top_patches``, and ``top_centers``.
        """
        from tools.seen_unseen_objaverse_general import _category_from_slug

        print("=" * 80)
        print("TEXT-TO-PATCH MATCHING")
        print("=" * 80)
        print(f"File:  {npz_file}")
        print(f"Query: '{query}'")

        # Load data
        npz_data = np.load(npz_file, allow_pickle=True)
        patch_feat = npz_data['patch_feat']
        patch_centers = npz_data['patch_centers']
        patch_indices = npz_data['patch_indices']
        points_sampled = npz_data['points_sampled']

        # Reconstruct patches
        num_patches = patch_indices.shape[0]
        points_per_patch = patch_indices.shape[1]
        patches = np.zeros((num_patches, points_per_patch, 3), dtype=np.float32)
        for i in range(num_patches):
            patches[i] = points_sampled[patch_indices[i]]

        _text_dim = patch_feat.shape[1]

        # Infer category from filename
        slug = os.path.basename(npz_data['meta_json'][0]['source_path']).replace('patches_', '')
        category = _category_from_slug(slug) if slug else None

        print(f"\nEncoding text query...")
        print(f"  Query:    '{query}'")
        print(f"  Category: {category}")

        # Encode text
        with torch.no_grad():
            text_feat = self.text_cache.encode_label_for_sample(query, category, setting="part_only", initial_texts=tuple(hints))
            text_feat = text_feat.to(self.device)

        print(f"✓ Text encoded: shape={text_feat.shape}, norm={torch.norm(text_feat).item():.4f}")

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
        top_indices = np.argsort(distances_np)[:top_k]

        print(f"\nTop {top_k} matches for '{query}':")
        print("─" * 80)
        for rank, idx in enumerate(top_indices[:3], 1):
            center = patch_centers[idx]
            print(f"  {rank:2d}. Patch {idx:3d}  |  sim={similarities_np[idx]:.4f}  |  "
                  f"center=[{center[0]:6.3f}, {center[1]:6.3f}, {center[2]:6.3f}]")

        if show_plots:
            self._plot_matches(
                points_sampled, patch_centers, patches,
                similarities_np, distances_np, top_indices, query
            )

        return {
            'similarities': similarities_np,
            'distances': distances_np,
            'top_indices': top_indices,
            'top_patches': patches[top_indices],
            'top_centers': patch_centers[top_indices]
        }

    def pipeline(
        self,
        input: str,
        query: str,
        ckpt: str = POINTBERT_CKPT,
        output: str | None = None,
        top_k: int = 10,
        show_plots: bool = True,
    ) -> Dict[str, Any]:
        """Full pipeline: extract patches then match to *query*.

        Args:
            input: Path to input file (``.ply``, ``.glb``, etc.).
            query: Text query (e.g. ``"nose"``, ``"wheel"``).
            ckpt: Path to checkpoint file.
            output: Output ``.npz`` file path (default: temp file).
            top_k: Number of top matches to return.
            show_plots: Whether to show matplotlib plots.

        Returns:
            Dictionary containing match results (see :meth:`match`).
        """
        # Extract patches
        npz_result = self.extract(input=input, ckpt=ckpt, output=output)
        npz_file: Union[str, BytesIO] = BytesIO(npz_result) if isinstance(npz_result, bytes) else npz_result

        # Match query
        results = self.match(npz_file=npz_file, query=query, top_k=top_k, show_plots=show_plots)

        return results

    def _plot_analysis(
        self,
        points_sampled: np.ndarray,
        patch_centers: np.ndarray,
        patches: np.ndarray,
        patch_emb: np.ndarray,
        patch_feat: np.ndarray | None,
    ) -> None:
        """Plot analysis visualisations: point cloud, patch centres, feature norms."""
        fig = plt.figure(figsize=(18, 6))

        # Full point cloud
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(
            points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],  # type: ignore[arg-type]
            c=points_sampled[:, 2], cmap='viridis', s=8, alpha=0.7
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Full Sampled Point Cloud\n(2048 points)', fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Z coordinate')

        # Point cloud with patch centers
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(
            points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],  # type: ignore[arg-type]
            c='lightgray', s=3, alpha=0.3, label='Points'
        )
        _scatter2 = ax2.scatter(
            patch_centers[:, 0], patch_centers[:, 1], patch_centers[:, 2],  # type: ignore[arg-type]
            c=np.arange(len(patch_centers)), cmap='tab20', s=40, alpha=0.9,
            edgecolors='black', linewidth=0.5, label='Patch Centers'
        )
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'Points with Patch Centers\n({len(patch_centers)} patches)', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)

        # Feature norms
        ax3 = fig.add_subplot(133)
        patch_emb_norms = np.linalg.norm(patch_emb, axis=1)
        x = np.arange(len(patch_emb_norms))
        ax3.plot(x, patch_emb_norms, 'o-', label='Point-BERT', alpha=0.7, markersize=3)
        if patch_feat is not None:
            patch_feat_norms = np.linalg.norm(patch_feat, axis=1)
            ax3.plot(x, patch_feat_norms, 's-', label='Projected Text', alpha=0.7, markersize=3)
        ax3.set_xlabel('Patch Index')
        ax3.set_ylabel('Feature Norm (L2)')
        ax3.set_title('Feature Magnitudes by Patch', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_matches(
        self,
        points_sampled: np.ndarray,
        patch_centers: np.ndarray,
        patches: np.ndarray,
        similarities_np: np.ndarray,
        distances_np: np.ndarray,
        top_indices: np.ndarray,
        query: str,
    ) -> None:
        """Plot matching visualisations: similarity map, top patches, histogram."""
        cmap = cm.get_cmap('RdYlGn')
        norm = Normalize(vmin=similarities_np.min(), vmax=similarities_np.max())

        # Figure 1: Overview
        fig1 = plt.figure(figsize=(20, 8))

        # Similarity map
        ax1 = fig1.add_subplot(131, projection='3d')
        ax1.scatter(
            points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],  # type: ignore[arg-type]
            c='lightgray', s=2, alpha=0.2
        )
        scatter1 = ax1.scatter(
            patch_centers[:, 0], patch_centers[:, 1], patch_centers[:, 2],  # type: ignore[arg-type]
            c=similarities_np, cmap='RdYlGn', s=80, alpha=0.8,
            edgecolors='black', linewidth=0.5
        )
        for i in range(min(3, len(top_indices))):
            idx = top_indices[i]
            ax1.scatter(
                patch_centers[idx, 0], patch_centers[idx, 1], patch_centers[idx, 2],
                c='gold', s=400, marker='*', edgecolors='darkred', linewidth=2, zorder=100
            )
        ax1.set_xlabel('X', fontweight='bold')
        ax1.set_ylabel('Y', fontweight='bold')
        ax1.set_zlabel('Z', fontweight='bold')
        ax1.set_title(f'Similarity to "{query}"', fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1, label='Similarity')

        # Top matches highlighted
        ax2 = fig1.add_subplot(132, projection='3d')
        ax2.scatter(
            points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],  # type: ignore[arg-type]
            c='lightgray', s=2, alpha=0.15
        )
        ax2.scatter(
            patch_centers[:, 0], patch_centers[:, 1], patch_centers[:, 2],  # type: ignore[arg-type]
            c='lightblue', s=40, alpha=0.3
        )
        colors = ['red', 'orange', 'gold', 'yellow', 'lightgreen']
        for i in range(min(5, len(top_indices))):
            idx = top_indices[i]
            patch_pts = patches[idx]
            ax2.scatter(
                patch_pts[:, 0], patch_pts[:, 1], patch_pts[:, 2],  # type: ignore[arg-type]
                c=colors[i], s=30, alpha=0.9, edgecolors='black', linewidth=0.5
            )
            ax2.scatter(
                patch_centers[idx, 0], patch_centers[idx, 1], patch_centers[idx, 2],
                c=colors[i], s=200, marker='*', edgecolors='black', linewidth=1.5, zorder=100
            )
        ax2.set_xlabel('X', fontweight='bold')
        ax2.set_ylabel('Y', fontweight='bold')
        ax2.set_zlabel('Z', fontweight='bold')
        ax2.set_title('Top 5 Matches', fontweight='bold')

        # Histogram
        ax3 = fig1.add_subplot(133)
        _n, bins, patches_hist = ax3.hist(
            similarities_np, bins=40, alpha=0.7, edgecolor='black', linewidth=1
        )
        for i, patch_rect in enumerate(patches_hist):  # type: ignore[arg-type]
            patch_rect.set_facecolor(cmap(norm(bins[i])))
        for i in range(min(3, len(top_indices))):
            ax3.axvline(
                similarities_np[top_indices[i]].item(), color='red',
                linestyle='--', linewidth=2, alpha=0.7
            )
        ax3.set_xlabel('Similarity Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Similarity Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'Text-to-Patch Matching: "{query}"', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Figure 2: Detailed patches
        fig2 = plt.figure(figsize=(20, 12))
        for i in range(min(6, len(top_indices))):
            idx = top_indices[i]
            patch_pts = patches[idx]
            center = patch_centers[idx]
            sim = similarities_np[idx]

            ax = fig2.add_subplot(2, 3, i+1, projection='3d')
            ax.scatter(
                patch_pts[:, 0], patch_pts[:, 1], patch_pts[:, 2],  # type: ignore[arg-type]
                c=patch_pts[:, 2], cmap='viridis', s=50, alpha=0.8,
                edgecolors='black', linewidth=0.3
            )
            ax.scatter(
                center[0], center[1], center[2],
                c='red', s=300, marker='*', edgecolors='darkred', linewidth=2, zorder=100
            )
            ax.set_xlabel('X', fontweight='bold')
            ax.set_ylabel('Y', fontweight='bold')
            ax.set_zlabel('Z', fontweight='bold')
            ax.set_title(
                f'Rank {i+1}: Patch {idx}\nSim: {sim:.4f}',
                fontweight='bold'
            )

        plt.suptitle(f'Top 6 Patches for "{query}"', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    """Main entry point"""
    fire.Fire(PatchExplorer)
