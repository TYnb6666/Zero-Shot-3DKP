"""Visualization commands - Plotting and visualization utilities"""
from pathlib import Path

import click

from zerokey._defaults import DEFAULT_LOG_DIR


@click.group(name='vis')
def vis_group() -> None:
    """Visualization and plotting utilities"""
    pass


@vis_group.command('gpt4o')
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
def gpt4o(image_path: Path) -> None:
    """GPT-4o point visualization demo\n\nIMAGE_PATH: path to the input image."""
    from zerokey.vis.gpt4o import main as gpt4o_main
    gpt4o_main(image_path)


@vis_group.command('describe')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='DescribePoints', show_default=True,
              help='Experiment name')
def describe(log_dir: Path, expname: str) -> None:
    """Saliency-driven point description and annotation"""
    from zerokey.vis.describe import Generator
    generator = Generator(log_dir, expname=expname)
    generator.main_loop()


@vis_group.command('schelling')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Output directory for results')
@click.option('--expname', '-e', default='SchellingAnnotate', show_default=True,
              help='Experiment name')
def schelling(log_dir: Path, expname: str) -> None:
    """Schelling point annotation and visualization"""
    from zerokey.vis.schelling import Generator
    generator = Generator(log_dir, expname=expname)
    generator.main_loop()


@vis_group.command('demo')
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
def demo(image_path: Path) -> None:
    """Simple demo for testing Molmo and GPT-4o\n\nIMAGE_PATH: path to the input image."""
    from zerokey.vis.demo import main as demo_main
    demo_main(image_path)
