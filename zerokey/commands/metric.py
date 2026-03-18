"""Metric commands - IoU calculation and evaluation utilities"""
from pathlib import Path

import click

from zerokey._defaults import DEFAULT_LOG_DIR


@click.group(name='metric')
def metric_group() -> None:
    """Calculate IoU and evaluation metrics"""
    pass


@metric_group.command('iou')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Directory containing results')
@click.option('--expname', '-e', required=True,
              help='Experiment name to evaluate')
@click.option('--ioref', '-r', multiple=True,
              help='Reference experiments for comparison (can specify multiple)')
def iou(log_dir: Path, expname: str, ioref: tuple[str, ...]) -> None:
    """Calculate IoU metrics for keypoint detection results

    Examples:
        zerokey metric iou --expname Rebuttal-1E
        zerokey metric iou --expname Rebuttal-1E --ioref Baseline1 --ioref Baseline2
    """
    from zerokey.io.kpnet import KPNetEvaluator
    evaluator = KPNetEvaluator(log_dir, expname=expname, ioref=ioref)
    evaluator.main_loop()


@metric_group.command('debug')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Directory containing results')
@click.option('--expname', '-e', required=True,
              help='Experiment name to evaluate')
def debug(log_dir: Path, expname: str) -> None:
    """Debug evaluation with per-mesh analysis"""
    from zerokey.io.debug import KPNetEvalDebug
    evaluator = KPNetEvalDebug(log_dir, expname=expname)
    evaluator.main_loop()


@metric_group.command('rawpts')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Directory containing results')
@click.option('--expname', '-e', required=True,
              help='Experiment name to evaluate')
@click.option('--gaussian-sigma', default=0.01, show_default=True,
              help='Gaussian sigma for smoothing')
def rawpts(log_dir: Path, expname: str, gaussian_sigma: float) -> None:
    """Raw point evaluation against ground truth"""
    from zerokey.io.rawpts import KPNetEvaluator
    evaluator = KPNetEvaluator(log_dir, expname=expname)
    evaluator.main_loop(gaussian_sigma=gaussian_sigma)


@metric_group.command('schelling')
@click.option('--log-dir', '-d', type=click.Path(path_type=Path),
              default=DEFAULT_LOG_DIR, show_default=True,
              help='Directory containing results')
@click.option('--expname', '-e', required=True,
              help='Experiment name to evaluate')
def schelling(log_dir: Path, expname: str) -> None:
    """Schelling dataset I/O (load/save keypoints)"""
    click.echo(f"SchellingIO initialized for {expname} at {log_dir}")
    click.echo("Use this class programmatically for saving/loading Schelling keypoints")
