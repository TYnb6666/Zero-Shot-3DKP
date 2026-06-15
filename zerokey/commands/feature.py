"""Feature-cache commands for ZeroKey experiments."""

from __future__ import annotations

from pathlib import Path

import click

from zerokey._defaults import DEFAULT_LOG_DIR
from zerokey.features.molmo_vit import FeatureDType, MolmoVitFeatureSaver


@click.group(name="feature")
def feature_group() -> None:
    """Save reusable intermediate feature caches."""


@feature_group.command(name="molmo-vit")
@click.option("--log-dir", "-d", type=click.Path(path_type=Path), default=DEFAULT_LOG_DIR, show_default=True, help="Root output directory")
@click.option("--expname", "-e", default="ZeroKey", show_default=True, help="Experiment name under log-dir")
@click.option("--classes", default="table,airplane,chair", show_default=True, help="Comma-separated KeypointNet class titles")
@click.option("--res", default=512, show_default=True, help="Base render resolution")
@click.option("--scale", default=2, show_default=True, help="Upscaling factor for high-res rendering")
@click.option("--use-texture/--no-texture", default=False, help="Use mesh textures for rendering")
@click.option("--model-path", default="allenai/Molmo-7B-D-0924", show_default=True, help="Molmo model identifier or local path")
@click.option("--output-name", default="molmo_vit_features.pt", show_default=True, help="Feature payload filename inside each mesh directory")
@click.option("--selected-layers", default="-10,-3", show_default=True, help="Comma-separated ViT layer indices to concatenate")
@click.option("--feature-dtype", type=click.Choice(["float32", "float16", "bfloat16", "keep"]), default="float16", show_default=True, help="Storage dtype for feature tensors; zbuf/cameras remain float32")
@click.option("--save-zbuf/--no-save-zbuf", default=True, show_default=True, help="Save nearest-depth z-buffer per view")
@click.option("--save-rgb-preview/--no-save-rgb-preview", default=False, show_default=True, help="Save per-view RGB preview PNGs")
@click.option("--include-skipped/--exclude-skipped", default=False, show_default=True, help="Include meshes listed in skipped_meshes.txt")
@click.option("--force/--no-force", default=False, show_default=True, help="Overwrite complete existing payloads")
@click.option("--max-meshes", type=int, default=0, show_default=True, help="Process at most N meshes after skip filtering (0 = all)")
@click.option("--max-retries", type=int, default=100, show_default=True, help="Retry attempts per incomplete/failed mesh")
@click.option("--retry-sleep", type=float, default=0.0, show_default=True, help="Seconds to sleep between retries")
@click.option("--batch-size", type=int, default=13, show_default=True, help="Rendering batch size for views")
def molmo_vit_cmd(
    log_dir: Path,
    expname: str,
    classes: str,
    res: int,
    scale: int,
    use_texture: bool,
    model_path: str,
    output_name: str,
    selected_layers: str,
    feature_dtype: FeatureDType,
    save_zbuf: bool,
    save_rgb_preview: bool,
    include_skipped: bool,
    force: bool,
    max_meshes: int,
    max_retries: int,
    retry_sleep: float,
    batch_size: int,
) -> None:
    """Save Molmo ViT global-crop feature maps for KeypointNet meshes."""
    class_list = [item.strip() for item in classes.split(",") if item.strip()]
    layer_list = [int(item.strip()) for item in selected_layers.split(",") if item.strip()]
    saver = MolmoVitFeatureSaver(
        log_dir=log_dir,
        expname=expname,
        res=res,
        scale=scale,
        use_texture=use_texture,
        model_path=model_path,
        output_name=output_name,
        selected_layers=layer_list,
        feature_dtype=feature_dtype,
        save_zbuf=save_zbuf,
        save_rgb_preview=save_rgb_preview,
        force=force,
        include_skipped=include_skipped,
        retry_sleep=retry_sleep,
    )
    saver.run(
        class_list,
        max_meshes=max_meshes if max_meshes > 0 else None,
        max_retries=max_retries,
        batch_size=batch_size,
    )
