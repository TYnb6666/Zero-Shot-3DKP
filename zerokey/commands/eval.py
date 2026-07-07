"""Evaluation command - Our ZeroKey method on different datasets"""
from pathlib import Path

import click

from zerokey._defaults import DEFAULT_LOG_DIR


@click.command(name='eval')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='ZeroKey', show_default=True,
              help='Experiment name')
@click.option('--dataset', type=click.Choice(['keypointnet', 'human3m', 'realscene']),
              default='keypointnet', show_default=True,
              help='Dataset to evaluate on')
@click.option('--use-texture/--no-texture', default=False,
              help='Use mesh textures for rendering')
@click.option('--res', default=512, show_default=True,
              help='Render resolution')
@click.option('--scale', default=2, show_default=True,
              help='Upscaling factor for high-res rendering')
@click.option('--num-shards', type=int, default=1, show_default=True,
              help='Total number of shards for KeypointNet class-wise splitting')
@click.option('--shard-id', type=int, default=0, show_default=True,
              help='0-based shard index for KeypointNet class-wise splitting')
@click.option('--max-meshes', type=int, default=0, show_default=True,
              help='Stop after processing N selected meshes (0 = no limit)')
@click.option('--use-molmo-vit-features/--no-molmo-vit-features', default=False, show_default=True,
              help='Use cached per-view Molmo ViT features during KeypointNet clustering')
@click.option('--molmo-vit-feature-dim', type=int, default=64, show_default=True,
              help='Random-projection dimension for cached Molmo ViT features before clustering')
@click.option('--molmo-vit-feature-scale', type=float, default=0.1, show_default=True,
              help='Scale applied to standardized reduced Molmo ViT features in HDBSCAN space')
@click.option('--molmo-vit-feature-file', default='molmo_vit_features.pt', show_default=True,
              help='Per-mesh cached Molmo ViT feature filename')
@click.option('--molmo-vit-feature-expname', default='', show_default=True,
              help='Source experiment name for cached Molmo ViT features (empty = current expname)')
@click.option('--molmo-2d-expname', default='', show_default=True,
              help='Source experiment name for cached raw Molmo 2D detections (empty = re-query Molmo)')
@click.option('--use-pointnext-surface-features/--no-pointnext-surface-features', default=False, show_default=True,
              help='Snap Molmo backprojected candidates to saved PointNeXt surface anchors before clustering')
@click.option('--pointnext-surface-feature-dir', type=click.Path(path_type=Path), default=None,
              help='Root directory containing PointNeXt surface artifacts as category/mesh_id.pt')
@click.option('--pointnext-feature-dim', type=int, default=16, show_default=True,
              help='Random-projection dimension for PointNeXt features before HDBSCAN')
@click.option('--pointnext-feature-scale', type=float, default=0.1, show_default=True,
              help='Scale applied to standardized PointNeXt features in HDBSCAN space')
@click.option('--pointnext-voxel-size', type=float, default=0.01, show_default=True,
              help='Voxel size for downsampling snapped candidates (<=0 disables)')
@click.option('--pointnext-max-disk-diameter', type=float, default=0.1, show_default=True,
              help='Reject a per-view 2D disk if its valid 3D backprojection robust lateral diameter is larger (<=0 disables)')
@click.option('--pointnext-max-disk-depth-range', type=float, default=0.03, show_default=True,
              help='Reject a per-view 2D disk if robust depth range along the viewing direction is larger (<=0 disables)')
@click.option('--pointnext-max-disk-depth-ratio', type=float, default=0.75, show_default=True,
              help='Reject a per-view 2D disk if depth range / lateral diameter is larger (<=0 disables)')
@click.option('--pointnext-min-disk-points', type=int, default=50, show_default=True,
              help='Minimum valid backprojected pixels required to keep a per-view disk before snapping')
@click.option('--pointnext-max-surface-dist', type=float, default=0.0, show_default=True,
              help='Optional max candidate-to-anchor distance after snapping (0 = disabled)')
def eval_cmd(log_dir: Path, expname: str, dataset: str, use_texture: bool, res: int, scale: int, num_shards: int, shard_id: int, max_meshes: int,
             use_molmo_vit_features: bool, molmo_vit_feature_dim: int, molmo_vit_feature_scale: float, molmo_vit_feature_file: str,
             molmo_vit_feature_expname: str, molmo_2d_expname: str, use_pointnext_surface_features: bool,
             pointnext_surface_feature_dir: Path | None, pointnext_feature_dim: int, pointnext_feature_scale: float,
             pointnext_voxel_size: float, pointnext_max_disk_diameter: float, pointnext_max_disk_depth_range: float,
             pointnext_max_disk_depth_ratio: float, pointnext_min_disk_points: int, pointnext_max_surface_dist: float) -> None:
    """Run ZeroKey 3D keypoint detection

    Examples:
        zerokey eval --expname Rebuttal-1E
        zerokey eval --dataset human3m --expname Human3M-Test
    """
    if dataset == 'keypointnet':
        from zerokey.generators.kpnet import KPNetGenerator
        generator = KPNetGenerator(
            log_dir,
            expname=expname,
            res=res,
            scale=scale,
            use_molmo_vit_features=use_molmo_vit_features,
            molmo_vit_feature_dim=molmo_vit_feature_dim,
            molmo_vit_feature_scale=molmo_vit_feature_scale,
            molmo_vit_feature_file=molmo_vit_feature_file,
            molmo_vit_feature_expname=molmo_vit_feature_expname or None,
            molmo_2d_expname=molmo_2d_expname or None,
            use_pointnext_surface_features=use_pointnext_surface_features,
            pointnext_surface_feature_dir=pointnext_surface_feature_dir,
            pointnext_feature_dim=pointnext_feature_dim,
            pointnext_feature_scale=pointnext_feature_scale,
            pointnext_voxel_size=pointnext_voxel_size,
            pointnext_max_disk_diameter=pointnext_max_disk_diameter,
            pointnext_max_disk_depth_range=pointnext_max_disk_depth_range,
            pointnext_max_disk_depth_ratio=pointnext_max_disk_depth_ratio,
            pointnext_min_disk_points=pointnext_min_disk_points,
            pointnext_max_surface_dist=pointnext_max_surface_dist if pointnext_max_surface_dist > 0 else None,
        )
        generator.main_loop(
            use_texture=use_texture,
            num_shards=num_shards,
            shard_id=shard_id,
            max_meshes=max_meshes if max_meshes > 0 else None,
        )
    elif dataset == 'human3m':
        from zerokey.generators.human3m import Human3MGenerator
        generator = Human3MGenerator(log_dir, expname=expname)
        generator.main_loop(use_texture=use_texture)
    elif dataset == 'realscene':
        from zerokey.generators.realscene import RealSceneGenerator
        generator = RealSceneGenerator(log_dir, expname=expname)
        generator.main_loop(use_texture=use_texture)
