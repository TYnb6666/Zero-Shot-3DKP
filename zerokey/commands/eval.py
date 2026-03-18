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
def eval_cmd(log_dir: Path, expname: str, dataset: str, use_texture: bool, res: int, scale: int) -> None:
    """Run ZeroKey 3D keypoint detection

    Examples:
        zerokey eval --expname Rebuttal-1E
        zerokey eval --dataset human3m --expname Human3M-Test
    """
    if dataset == 'keypointnet':
        from zerokey.generators.kpnet import KPNetGenerator
        generator = KPNetGenerator(log_dir, expname=expname, res=res, scale=scale)
        generator.main_loop(use_texture=use_texture)
    elif dataset == 'human3m':
        from zerokey.generators.human3m import Human3MGenerator
        generator = Human3MGenerator(log_dir, expname=expname)
        generator.main_loop(use_texture=use_texture)
    elif dataset == 'realscene':
        from zerokey.generators.realscene import RealSceneGenerator
        generator = RealSceneGenerator(log_dir, expname=expname)
        generator.main_loop(use_texture=use_texture)
