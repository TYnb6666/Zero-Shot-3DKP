"""Data preparation commands - sample and render KeypointNet datasets."""
from pathlib import Path

import click

from zerokey._defaults import KEYPOINT_DATASET_PATH


@click.group(name='data')
def data_group() -> None:
    """Data preparation pipeline for KeypointNet."""
    pass


@data_group.command('sample')
@click.option('--save-dir', type=click.Path(path_type=Path), required=True,
              help='Directory to save the dataset (reuse for render)')
@click.option('--keypointnet-dir', type=click.Path(path_type=Path),
              default=KEYPOINT_DATASET_PATH, show_default=True,
              help='KeypointNet dataset directory')
@click.option('--splits', multiple=True, default=['train', 'val', 'test'],
              show_default=True, help='Dataset splits')
@click.option('--max-shapes-per-split', type=int, default=160,
              show_default=True, help='Max shapes per split')
@click.option('--seed', type=int, default=2024, show_default=True,
              help='Random seed')
def sample(save_dir: Path, keypointnet_dir: Path, splits: tuple[str, ...],
           max_shapes_per_split: int, seed: int) -> None:
    """Sample shapes from KeypointNet splits.

    Examples:
        zerokey data sample --save-dir ./rendered --keypointnet-dir ./keypointnet
    """
    import argparse
    from data_creation.keypointnet.sample_dataset import sample_shapes

    args = argparse.Namespace(
        save_dir=str(save_dir),
        keypointnet_dir=str(keypointnet_dir),
        splits=list(splits),
        max_shapes_per_split=max_shapes_per_split,
        seed=seed,
    )
    sample_shapes(args, save=True)


@data_group.command('render')
@click.option('--save-dir', type=click.Path(path_type=Path), required=True,
              help='Directory to save the dataset (same as sample)')
@click.option('--keypointnet-dir', type=click.Path(path_type=Path),
              default=KEYPOINT_DATASET_PATH, show_default=True,
              help='KeypointNet dataset directory')
@click.option('--splits', multiple=True, default=['train', 'val', 'test'],
              show_default=True, help='Dataset splits')
@click.option('--seed', type=int, default=2024, show_default=True,
              help='Random seed')
@click.option('--n-points', type=int, default=1024, show_default=True,
              help='Number of sampled points')
@click.option('--render-res', type=int, default=512, show_default=True,
              help='Render resolution')
@click.option('--num-random-views', type=int, default=108, show_default=True,
              help='Number of random views')
@click.option('--n-ring', type=int, default=10, show_default=True,
              help='Number of rings')
def render(save_dir: Path, keypointnet_dir: Path, splits: tuple[str, ...],
           seed: int, n_points: int, render_res: int, num_random_views: int,
           n_ring: int) -> None:
    """Render sampled shapes (run after 'sample').

    Examples:
        zerokey data render --save-dir ./rendered --keypointnet-dir ./keypointnet
    """
    import argparse
    from data_creation.keypointnet.render_dataset import main

    args = argparse.Namespace(
        save_dir=str(save_dir),
        keypointnet_dir=str(keypointnet_dir),
        splits=list(splits),
        seed=seed,
        n_points=n_points,
        render_res=render_res,
        num_random_views=num_random_views,
        n_ring=n_ring,
    )
    main(args)
