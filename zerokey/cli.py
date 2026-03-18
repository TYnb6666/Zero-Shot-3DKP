"""ZeroKey CLI - Unified command-line interface for Zero-Shot 3D Keypoint Detection"""
import click


@click.group()
@click.version_option(version="0.1.0", prog_name="zerokey")
def main() -> None:
    """ZeroKey: Zero-Shot 3D Keypoint Detection from Large Language Models

    A CLI for running evaluations, baselines, metrics, and visualizations.
    """
    pass


# Import and register command groups
from zerokey.commands import eval as eval_cmd
from zerokey.commands import baseline as baseline_cmd
from zerokey.commands import metric as metric_cmd
from zerokey.commands import vis as vis_cmd
from zerokey.commands import data as data_cmd

main.add_command(eval_cmd.eval_cmd, name='eval')
main.add_command(baseline_cmd.baseline_group, name='baseline')
main.add_command(metric_cmd.metric_group, name='metric')
main.add_command(vis_cmd.vis_group, name='vis')
main.add_command(data_cmd.data_group, name='data')


if __name__ == '__main__':
    main()
