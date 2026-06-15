"""Persist Molmo ViT patch feature maps for rendered KeypointNet meshes.

The saver is intentionally resumable: each mesh writes to a temporary file first,
validates the payload, and then atomically renames it to the final
``molmo_vit_features.pt`` path.  Existing final files are loaded and validated
before they are treated as complete.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

FeatureDType = Literal["float32", "float16", "bfloat16", "keep"]


@dataclass(frozen=True)
class MeshFeatureStatus:
    """Validation result for one saved Molmo feature payload."""

    complete: bool
    reason: str = ""
    payload: dict[str, Any] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class MeshFeatureRecord:
    """One KeypointNet mesh selected for Molmo feature caching."""

    class_title: str
    mesh_id: str
    dataset_index: int


class MolmoVitFeatureSaver:
    """Save Molmo ViT global-crop feature maps for KeypointNet meshes.

    Each final payload contains one tensor per rendered view:

    * ``features``: ``[num_views, 24, 24, 2048]`` ViT layer-concat features.
    * ``zbuf``: optional ``[num_views, H, W]`` nearest-surface render depth.
    * ``camera_R`` / ``camera_T`` / ``camera_center``: camera tensors aligned
      with view indices.
    * ``metadata``: shape, timing, crop, and rendering information.
    """

    def __init__(
        self,
        log_dir: str | os.PathLike[str],
        expname: str,
        *,
        res: int = 512,
        scale: int = 2,
        use_texture: bool = False,
        model_path: str = "allenai/Molmo-7B-D-0924",
        output_name: str = "molmo_vit_features.pt",
        selected_layers: Sequence[int] = (-10, -3),
        feature_dtype: FeatureDType = "float16",
        save_zbuf: bool = True,
        save_rgb_preview: bool = False,
        force: bool = False,
        include_skipped: bool = False,
        retry_sleep: float = 0.0,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.expname = expname
        self.output_root = self.log_dir / self.expname
        self.res = res
        self.scale = scale
        self.use_texture = use_texture
        self.model_path = model_path
        self.output_name = output_name
        self.selected_layers = tuple(selected_layers)
        self.feature_dtype = feature_dtype
        self.save_zbuf = save_zbuf
        self.save_rgb_preview = save_rgb_preview
        self.force = force
        self.include_skipped = include_skipped
        self.retry_sleep = retry_sleep

        self._generator: Any | None = None
        self._molmo: Any | None = None

    @property
    def generator(self) -> Any:
        """Lazily create the ZeroKey renderer/generator."""
        if self._generator is None:
            from zerokey.generators.kpnet import KPNetGenerator

            self._generator = KPNetGenerator(
                log_dir=self.log_dir,
                expname=self.expname,
                res=self.res,
                scale=self.scale,
            )
        return self._generator

    @property
    def molmo(self) -> Any:
        """Lazily create the Molmo model used for ViT feature extraction."""
        if self._molmo is None:
            from zerokey.models.molmo import Molmo

            self._molmo = Molmo(model_path=self.model_path)
        return self._molmo

    def output_path(self, class_title: str, mesh_id: str) -> Path:
        """Return the final feature payload path for one mesh."""
        return self.output_root / class_title / mesh_id / self.output_name

    def manifest_path(self, class_title: str, mesh_id: str) -> Path:
        """Return the sidecar JSON manifest path for one mesh."""
        return self.output_path(class_title, mesh_id).with_name("molmo_vit_features_manifest.json")

    def iter_records(self, classes: Sequence[str], max_meshes: int | None = None) -> Iterable[MeshFeatureRecord]:
        """Yield KeypointNet records in deterministic dataset order."""
        from kp_utils import KeypointNetDataset

        yielded = 0
        skipped = self._load_skipped_meshes()
        for class_title in classes:
            dataset = KeypointNetDataset(filter_classes=[class_title], use_texture=self.use_texture)
            for idx in range(len(dataset)):
                _class_id, mesh_id = dataset.get_class_and_mesh_id(idx)
                record = MeshFeatureRecord(class_title=class_title, mesh_id=str(mesh_id), dataset_index=idx)
                if not self.include_skipped and (record.class_title, record.mesh_id) in skipped:
                    continue
                yield record
                yielded += 1
                if max_meshes is not None and yielded >= max_meshes:
                    return

    def run(
        self,
        classes: Sequence[str],
        *,
        max_meshes: int | None = None,
        max_retries: int = 100,
        batch_size: int | None = 13,
    ) -> dict[str, int]:
        """Save features for selected meshes with validation and retry.

        Args:
            classes: Class titles to process.
            max_meshes: Optional global limit after skip-list filtering.
            max_retries: Number of attempts per incomplete mesh.
            batch_size: View rendering batch size.
        """
        stats = {"selected": 0, "complete": 0, "skipped_complete": 0, "failed": 0}
        records = list(self.iter_records(classes, max_meshes=max_meshes))
        total = len(records)
        print(f"[INFO] selected_records={total} classes={','.join(classes)}")
        print(f"[INFO] output_root={self.output_root}")
        print(f"[INFO] max_retries={max_retries} feature_dtype={self.feature_dtype} save_zbuf={self.save_zbuf}")

        for order, record in enumerate(records, start=1):
            stats["selected"] += 1
            print("-" * 100)
            print(f"[MESH {order}/{total}] {record.class_title} {record.mesh_id}")

            if not self.force:
                status = self.validate_file(self.output_path(record.class_title, record.mesh_id))
                if status.complete:
                    stats["complete"] += 1
                    stats["skipped_complete"] += 1
                    print(f"[SKIP] complete existing payload: {self.output_path(record.class_title, record.mesh_id)}")
                    continue
                if status.reason:
                    print(f"[INFO] existing payload is incomplete/corrupt: {status.reason}")

            success = False
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"[ATTEMPT {attempt}/{max_retries}] {record.class_title} {record.mesh_id}")
                    self.save_one(record, batch_size=batch_size)
                    status = self.validate_file(self.output_path(record.class_title, record.mesh_id))
                    if not status.complete:
                        raise RuntimeError(f"post-save validation failed: {status.reason}")
                    stats["complete"] += 1
                    success = True
                    print(f"[DONE] {record.class_title} {record.mesh_id}")
                    break
                except Exception as exc:
                    print(f"[WARN] attempt {attempt}/{max_retries} failed for {record.class_title} {record.mesh_id}: {exc}")
                    self._cleanup_tmp_files(record.class_title, record.mesh_id)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if attempt < max_retries and self.retry_sleep > 0:
                        time.sleep(self.retry_sleep)
            if not success:
                stats["failed"] += 1
                self._record_failure(record, f"failed after {max_retries} attempts")

        print("=" * 100)
        print(f"[SUMMARY] {stats}")
        return stats

    @torch.inference_mode()
    def save_one(self, record: MeshFeatureRecord, *, batch_size: int | None = 13) -> Path:
        """Render one mesh, extract Molmo ViT features, and atomically save it."""
        from kp_utils import KeypointNetDataset

        t_total0 = time.perf_counter()
        out_path = self.output_path(record.class_title, record.mesh_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
        manifest_path = self.manifest_path(record.class_title, record.mesh_id)

        dataset = KeypointNetDataset(filter_classes=[record.class_title], use_texture=self.use_texture)
        mesh, _keypoints, class_title, mesh_id, _pcd = dataset[record.dataset_index]
        if class_title != record.class_title or mesh_id != record.mesh_id:
            mesh, class_title, mesh_id = self._find_mesh(record, dataset)

        gen = self.generator
        t_render0 = time.perf_counter()
        images, fragments, cameras, _lights = gen.views_from_model(
            mesh,
            gen.views,
            batch_size=batch_size,
            device=gen.device,
        )
        t_render1 = time.perf_counter()
        rendered_shape = tuple(images.shape)
        zbuf_shape = tuple(fragments.zbuf.shape)

        if gen.scale != 1:
            images = F.interpolate(images, scale_factor=1 / gen.scale, mode="bicubic", align_corners=False)
        if images.dtype != torch.uint8 and images.ndim == 4:
            images = (images * 255).clamp_(0, 255).to(torch.uint8)
        molmo_image_shape = tuple(images.shape)

        feature_maps: list[torch.Tensor] = []
        crop_counts: list[int] = []
        view_times: list[float] = []
        t_features0 = time.perf_counter()
        for view_idx in range(images.shape[0]):
            view_t0 = time.perf_counter()
            image = self._tensor_view_to_pil(images[view_idx])
            feature_map, num_crops = self.extract_global_crop_feature(image)
            feature_maps.append(self._cast_feature(feature_map.detach().cpu()))
            crop_counts.append(num_crops)
            view_times.append(time.perf_counter() - view_t0)
            print(
                f"[VIEW {view_idx:02d}/{images.shape[0]-1:02d}] "
                f"feature={tuple(feature_maps[-1].shape)} dtype={feature_maps[-1].dtype} "
                f"num_crops={num_crops} time={view_times[-1]:.3f}s"
            )
        t_features1 = time.perf_counter()
        features = torch.stack(feature_maps, dim=0)

        camera_r = cameras.R.detach().cpu().float()
        camera_t = cameras.T.detach().cpu().float()
        camera_center = cameras.get_camera_center().detach().cpu().float()
        zbuf = fragments.zbuf[..., 0].detach().cpu().float() if self.save_zbuf else None

        metadata: dict[str, Any] = {
            "complete": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "class_title": record.class_title,
            "mesh_id": record.mesh_id,
            "dataset_index": record.dataset_index,
            "model_path": self.model_path,
            "selected_layers": list(self.selected_layers),
            "used_crop_index": 0,
            "used_crop_type": "global",
            "processor_num_crops_per_view": crop_counts,
            "num_views": int(features.shape[0]),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype).removeprefix("torch."),
            "zbuf_shape": list(zbuf.shape) if zbuf is not None else None,
            "zbuf_dtype": str(zbuf.dtype).removeprefix("torch.") if zbuf is not None else None,
            "camera_R_shape": list(camera_r.shape),
            "camera_T_shape": list(camera_t.shape),
            "camera_center_shape": list(camera_center.shape),
            "render_res": self.res,
            "scale": self.scale,
            "use_texture": self.use_texture,
            "rendered_image_shape_before_downsample": list(rendered_shape),
            "fragments_zbuf_shape": list(zbuf_shape),
            "molmo_image_shape": list(molmo_image_shape),
            "patch_grid": [int(features.shape[1]), int(features.shape[2])],
            "feature_dim": int(features.shape[-1]),
            "timing_sec": {
                "render": t_render1 - t_render0,
                "features_total": t_features1 - t_features0,
                "features_mean_per_view": float(sum(view_times) / max(len(view_times), 1)),
                "total_before_save": time.perf_counter() - t_total0,
            },
        }

        payload: dict[str, Any] = {
            "features": features,
            "camera_R": camera_r,
            "camera_T": camera_t,
            "camera_center": camera_center,
            "metadata": metadata,
        }
        if zbuf is not None:
            payload["zbuf"] = zbuf

        if self.save_rgb_preview:
            self._save_rgb_preview(images, out_path.parent)

        t_save0 = time.perf_counter()
        torch.save(payload, tmp_path)
        status = self.validate_file(tmp_path)
        if not status.complete:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"temporary payload validation failed: {status.reason}")
        os.replace(tmp_path, out_path)
        t_save1 = time.perf_counter()

        metadata["timing_sec"]["save"] = t_save1 - t_save0
        metadata["timing_sec"]["total"] = time.perf_counter() - t_total0
        metadata["file_size_bytes"] = out_path.stat().st_size
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVE] {out_path} ({out_path.stat().st_size / 1024**2:.2f} MB)")
        return out_path

    @torch.inference_mode()
    def extract_global_crop_feature(self, image: Image.Image) -> tuple[torch.Tensor, int]:
        """Extract ``[24, 24, 2048]`` Molmo ViT features from crop 0."""
        molmo = self.molmo
        processor = molmo.processor
        model = molmo.model
        inputs = processor.process(images=[image], text="Describe this image.")
        inputs = {key: value.to(model.device).unsqueeze(0) for key, value in inputs.items()}
        if "images" not in inputs:
            raise RuntimeError("Molmo processor did not return an images tensor")
        inputs["images"] = inputs["images"].to(model.dtype)

        processed_images = inputs["images"]
        if processed_images.ndim != 4:
            raise RuntimeError(f"expected processor images [B,T,N,D], got {tuple(processed_images.shape)}")
        _batch, num_crops, num_patches, _pixels = processed_images.shape
        global_crop = processed_images[:, 0, :, :]

        vision_backbone = model.model.vision_backbone
        if vision_backbone is None:
            raise RuntimeError("Molmo model has no vision_backbone")
        hidden_states = vision_backbone.image_vit(global_crop)
        num_layers = len(hidden_states)
        selected: list[torch.Tensor] = []
        for layer in self.selected_layers:
            layer_idx = num_layers + layer if layer < 0 else layer
            if not 0 <= layer_idx < num_layers:
                raise IndexError(f"selected ViT layer {layer} resolved to {layer_idx}, outside 0..{num_layers - 1}")
            selected.append(hidden_states[layer_idx])
        features = torch.cat(selected, dim=-1)

        total_tokens = features.shape[1]
        if total_tokens == num_patches + 1:
            features = features[:, 1:, :]
        elif total_tokens != num_patches:
            raise RuntimeError(f"unexpected token count {total_tokens}; expected {num_patches} or {num_patches + 1}")

        grid = int(num_patches ** 0.5)
        if grid * grid != num_patches:
            raise RuntimeError(f"cannot reshape {num_patches} patches into a square grid")
        return features.reshape(grid, grid, features.shape[-1]), int(num_crops)

    def validate_file(self, path: Path) -> MeshFeatureStatus:
        """Load and validate a saved payload before accepting progress."""
        if not path.exists():
            return MeshFeatureStatus(False, "missing")
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            return MeshFeatureStatus(False, f"load failed: {exc}")
        if not isinstance(payload, dict):
            return MeshFeatureStatus(False, "payload is not a dict")

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("complete") is not True:
            return MeshFeatureStatus(False, "metadata.complete is not true")

        features = payload.get("features")
        if not isinstance(features, torch.Tensor):
            return MeshFeatureStatus(False, "features tensor missing")
        if features.ndim != 4 or tuple(features.shape[1:3]) != (24, 24):
            return MeshFeatureStatus(False, f"unexpected features shape {tuple(features.shape)}")
        if features.shape[-1] != 2048:
            return MeshFeatureStatus(False, f"unexpected feature dim {features.shape[-1]}")
        num_views = int(features.shape[0])
        if num_views <= 0:
            return MeshFeatureStatus(False, "features has no views")
        expected_dtype = self._expected_feature_torch_dtype()
        if expected_dtype is not None and features.dtype != expected_dtype:
            return MeshFeatureStatus(False, f"features dtype {features.dtype} != expected {expected_dtype}")
        if not torch.isfinite(features.float()).all().item():
            return MeshFeatureStatus(False, "features contains non-finite values")

        meta_num_views = metadata.get("num_views")
        if meta_num_views is not None and int(meta_num_views) != num_views:
            return MeshFeatureStatus(False, "metadata num_views does not match tensor")

        for key, shape_tail in (("camera_R", (3, 3)), ("camera_T", (3,)), ("camera_center", (3,))):
            tensor = payload.get(key)
            if not isinstance(tensor, torch.Tensor):
                return MeshFeatureStatus(False, f"{key} tensor missing")
            if tensor.shape[0] != num_views or tuple(tensor.shape[1:]) != shape_tail:
                return MeshFeatureStatus(False, f"unexpected {key} shape {tuple(tensor.shape)}")
            if not torch.isfinite(tensor.float()).all().item():
                return MeshFeatureStatus(False, f"{key} contains non-finite values")

        zbuf = payload.get("zbuf")
        if self.save_zbuf:
            if not isinstance(zbuf, torch.Tensor):
                return MeshFeatureStatus(False, "zbuf tensor missing")
            if zbuf.ndim != 3 or zbuf.shape[0] != num_views:
                return MeshFeatureStatus(False, f"unexpected zbuf shape {tuple(zbuf.shape)}")
            if zbuf.dtype != torch.float32:
                return MeshFeatureStatus(False, f"zbuf dtype {zbuf.dtype} != expected torch.float32")
            finite_or_background = torch.isfinite(zbuf.float()) | (zbuf.float() < 0)
            if not finite_or_background.all().item():
                return MeshFeatureStatus(False, "zbuf contains invalid values")

        meta_shape = metadata.get("feature_shape")
        if meta_shape is not None and list(features.shape) != list(meta_shape):
            return MeshFeatureStatus(False, "metadata feature_shape does not match tensor")
        return MeshFeatureStatus(True, payload=payload)

    def _find_mesh(self, record: MeshFeatureRecord, dataset: Any) -> tuple[Any, str, str]:
        """Fallback lookup by mesh id when a cached dataset index no longer matches."""
        for idx in range(len(dataset)):
            mesh, _keypoints, class_title, mesh_id, _pcd = dataset[idx]
            if class_title == record.class_title and mesh_id == record.mesh_id:
                return mesh, class_title, mesh_id
        raise RuntimeError(f"cannot find mesh {record.class_title} {record.mesh_id}")

    def _expected_feature_torch_dtype(self) -> torch.dtype | None:
        if self.feature_dtype == "float32":
            return torch.float32
        if self.feature_dtype == "float16":
            return torch.float16
        if self.feature_dtype == "bfloat16":
            return torch.bfloat16
        return None

    def _cast_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if self.feature_dtype == "float32":
            return feature.float()
        if self.feature_dtype == "float16":
            return feature.to(torch.float16)
        if self.feature_dtype == "bfloat16":
            return feature.to(torch.bfloat16)
        return feature

    @staticmethod
    def _tensor_view_to_pil(tensor_image: torch.Tensor) -> Image.Image:
        array = rearrange(tensor_image, "c h w -> h w c").detach().cpu().numpy()
        image = Image.fromarray(array)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _load_skipped_meshes(self) -> set[tuple[str, str]]:
        path = self.output_root / "skipped_meshes.txt"
        skipped: set[tuple[str, str]] = set()
        if not path.exists():
            return skipped
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                skipped.add((parts[0], parts[1]))
        return skipped

    def _cleanup_tmp_files(self, class_title: str, mesh_id: str) -> None:
        out_path = self.output_path(class_title, mesh_id)
        for tmp_path in out_path.parent.glob(f".{out_path.name}.tmp.*"):
            tmp_path.unlink(missing_ok=True)

    def _record_failure(self, record: MeshFeatureRecord, reason: str) -> None:
        path = self.output_root / "molmo_vit_feature_failures.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(
                "\t".join(
                    [
                        record.class_title,
                        record.mesh_id,
                        datetime.now(timezone.utc).isoformat(),
                        reason.replace("\n", " "),
                    ]
                )
                + "\n"
            )

    @staticmethod
    def _save_rgb_preview(images: torch.Tensor, out_dir: Path) -> None:
        preview_dir = out_dir / "molmo_vit_feature_views"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for view_idx in range(images.shape[0]):
            image = MolmoVitFeatureSaver._tensor_view_to_pil(images[view_idx])
            image.save(preview_dir / f"view_{view_idx:02d}.png")
