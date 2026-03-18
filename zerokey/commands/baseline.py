"""Baseline commands - Comparison methods"""
from pathlib import Path

import click

from zerokey._defaults import DEFAULT_LOG_DIR


@click.group(name='baseline')
def baseline_group() -> None:
    """Run baseline comparison methods"""
    pass


@baseline_group.command('patchalign3d')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='PatchAlign3D', show_default=True,
              help='Experiment name')
@click.option('--save-images/--no-save-images', default=False,
              help='Save rendered images')
def patchalign3d(log_dir: Path, expname: str, save_images: bool) -> None:
    """PatchAlign3D baseline (Point-BERT patch alignment)"""
    from zerokey.generators.patchalign3d import PatchAlign3DGenerator
    generator = PatchAlign3DGenerator(log_dir, expname=expname)
    generator.main_loop(save_rendered_images=save_images)


@baseline_group.command('patchalign3dzerokey')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='PatchAlign3DZeroKey', show_default=True,
              help='Experiment name')
@click.option('--save-images/--no-save-images', default=False,
              help='Save rendered images')
def patchalign3dzerokey(log_dir: Path, expname: str, save_images: bool) -> None:
    """PatchAlign3DZeroKey baseline"""
    from zerokey.generators.patchalign3dzerokey import PatchAlign3DZeroKeyGenerator
    generator = PatchAlign3DZeroKeyGenerator(log_dir, expname=expname)
    generator.main_loop(save_rendered_images=save_images)


@baseline_group.command('patchalign3dref')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='PatchAlign3DRef', show_default=True,
              help='Experiment name')
def patchalign3dref(log_dir: Path, expname: str) -> None:
    """PatchAlign3D with reference view support"""
    from zerokey.generators.patchalign3dref import PatchAlign3DRefGenerator
    generator = PatchAlign3DRefGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('ulip2ref')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='ULIP2Ref', show_default=True,
              help='Experiment name')
def ulip2ref(log_dir: Path, expname: str) -> None:
    """ULIP2 reference view evaluation"""
    from zerokey.generators.ulip2ref import ULIP2RefGenerator
    generator = ULIP2RefGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('paligemma')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='PaliGemma', show_default=True,
              help='Experiment name')
def paligemma(log_dir: Path, expname: str) -> None:
    """PaliGemma MLLM evaluation"""
    from zerokey.generators.paligemma import PaliGemmaGenerator
    generator = PaliGemmaGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('clip-dinoiser')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='CLIPDinoiser', show_default=True,
              help='Experiment name')
def clip_dinoiser(log_dir: Path, expname: str) -> None:
    """CLIP-DINOiser semantic segmentation baseline"""
    from zerokey.generators.clip_dinoiser import ClipDINOiserGenerator
    generator = ClipDINOiserGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('redcircle')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='RedCircle', show_default=True,
              help='Experiment name')
def redcircle(log_dir: Path, expname: str) -> None:
    """Red circle heuristic baseline"""
    from zerokey.generators.redcircle import RedCircleGenerator
    generator = RedCircleGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('saliency')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='Saliency', show_default=True,
              help='Experiment name')
def saliency(log_dir: Path, expname: str) -> None:
    """Saliency-based baseline using DINOv2"""
    from zerokey.generators.saliency import SaliencyGenerator
    generator = SaliencyGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('stable-keypoints')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='StableKeypoints', show_default=True,
              help='Experiment name')
def stable_keypoints(log_dir: Path, expname: str) -> None:
    """Unsupervised stable keypoints baseline"""
    from zerokey.generators.stable_keypoints import StableKeypoints
    generator = StableKeypoints(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('gpt4o')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='GPT4o', show_default=True,
              help='Experiment name')
def gpt4o(log_dir: Path, expname: str) -> None:
    """GPT-4o localization baseline"""
    from zerokey.generators.gpt4o import GPT4oGenerator
    generator = GPT4oGenerator(log_dir, expname=expname)
    generator.main_loop()


@baseline_group.command('bt3d')
@click.option('--filename', '-f', default='experiment.txt', show_default=True,
              help='Output filename for results')
@click.option('--use-texture/--no-texture', default=False,
              help='Use mesh textures')
@click.option('--model', type=click.Choice(['dino', 'clip', 'sam']),
              default='dino', show_default=True,
              help='Feature extraction model')
@click.option('--gaussian-sigma', default=0.01, show_default=True,
              help='Gaussian sigma for smoothing')
def bt3d(filename: str, use_texture: bool, model: str, gaussian_sigma: float) -> None:
    """BT3D benchmark evaluation (few-shot feature learning)"""
    from zerokey.generators.bt3d import main as bt3d_main
    bt3d_main(filename=filename, use_texture=use_texture,
              use_model=model, gaussian_sigma=gaussian_sigma)
