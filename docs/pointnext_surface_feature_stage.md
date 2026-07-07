# PointNeXt Surface Feature Extraction Stage

This note specifies the first-stage offline artifact for adding PointNeXt 3D surface features to ZeroKey candidate aggregation. The stage produces one reusable `.pt` file per KeypointNet mesh. Later candidate snapping must use the saved anchors in that file, not a newly sampled point cloud.

## Goal and invariants

The input to PointNeXt is the clean mesh surface, not ground-truth keypoints, Molmo 2D predictions, or Molmo backprojected candidates. For each mesh, uniformly sample a dense surface pool, run farthest-point sampling (FPS) to select exactly 4096 surface anchors, extract a per-anchor PointNeXt feature on those same anchors, and save the anchor coordinates plus aligned features.

Required invariants:

1. `surface_xyz[i]` and `surface_feat[i]` describe the same fixed surface anchor.
2. Future Molmo candidates must snap only to `surface_xyz` from the saved `.pt` file.
3. No later stage may regenerate a different surface cloud for snapping or feature assignment.
4. `surface_xyz` must be in the same object coordinate frame as ZeroKey backprojected `candidate_xyz`.
5. Dense intermediate samples are not saved in the standard artifact.

## Offline processing flow

For each KeypointNet mesh:

```text
mesh in ZeroKey coordinates
  -> deterministic uniform surface sampling, dense_n = 20000
  -> deterministic FPS, fps_n = 4096
  -> compute normals for the selected anchors
  -> PointNeXt-S C=64 ShapeNetPart part-segmentation inference
  -> capture decoder-final pre-segmentation-head feature, [4096, D]
  -> save category/<mesh_id>.pt
```

### Coordinate-frame policy

The extraction script must load and normalize meshes through the same mesh-loading path used by ZeroKey rendering/backprojection. If any transform is applied before rendering, the same transform must be applied before surface sampling. The saved `norm_info` is descriptive metadata for audits; it is not permission for downstream code to re-normalize candidates independently.

Recommended sanity checks per mesh:

- Compare saved `bbox_min` and `bbox_max` against the mesh passed to ZeroKey rendering.
- Render a small subset of `surface_xyz` with the same camera used by ZeroKey and verify that projected anchors lie on the rendered object silhouette.
- During candidate snapping integration, log the distribution of nearest-anchor distances; unexpectedly large distances usually indicate a coordinate mismatch or off-surface backprojection artifacts.

### Sampling and determinism

Use a deterministic seed derived from a global seed and a stable mesh identifier, or store the actual seed used per mesh. The saved `.pt` file is the source of truth after extraction, so deterministic sampling is mainly needed for reproducibility and recovery if artifacts are regenerated.

Uniform surface sampling should be area-weighted over mesh triangles. FPS should run on the dense sampled xyz coordinates and return 4096 selected anchors. If normals are used by the PointNeXt checkpoint, select the corresponding dense normals with the same FPS indices.

## Recommended `.pt` schema

The default artifact should save only fields required for downstream candidate snapping, feature assignment, reproducibility, and coordinate audits:

```python
{
    "schema_version": "pointnext_surface_features_v1",
    "mesh_id": str,
    "category": str,

    # Required downstream fields.
    "surface_xyz": torch.FloatTensor,       # [4096, 3], float32, ZeroKey object coordinates
    "surface_feat": torch.FloatTensor,      # [4096, D], float32, aligned with surface_xyz

    # Recommended for PointNeXt reproducibility and downstream geometric filters.
    "surface_normal": torch.FloatTensor,    # [4096, 3], float32, same row order as surface_xyz

    "sampling_info": {
        "dense_n": 20000,
        "fps_n": 4096,
        "sampling_method": "area_uniform_surface_sampling_then_fps",
        "seed": int,
        "deterministic": True,
        "saved_dense_points": False,
    },

    "model_info": {
        "backbone": "PointNeXt-S",
        "width": 64,
        "checkpoint": str,
        "checkpoint_sha256": str | None,
        "pretrained_dataset": "ShapeNetPart",
        "task": "part_segmentation",
        "feature_layer": "decoder_final_before_seg_head",
        "feature_dim": int,
        "input_points": 4096,
        "input_channels": 6,
        "use_normal": True,
        "feature_dtype": "float32",
    },

    "norm_info": {
        "coordinate_system": "same_as_zerokey_backproject_xyz",
        "mesh_loader": str,
        "applied_transform": str,
        "bbox_min": torch.FloatTensor,      # [3], float32
        "bbox_max": torch.FloatTensor,      # [3], float32
        "bbox_diag": float,
    },
}
```

`fps_idx` is optional and should be omitted in the default storage-efficient artifact because it only indexes the discarded dense pool. If a debug mode is added, it may save `fps_idx`, `dense_xyz`, and `dense_normal` into a separate debug output directory rather than into production artifacts.

## Output layout

Use one file per mesh under a checkpoint- and sampling-specific root:

```text
/data/taoye/zero-shot/features/pointnext_shapenetpart_c64_4096/
├── airplane/
│   ├── <mesh_id>.pt
├── chair/
│   ├── <mesh_id>.pt
└── table/
    ├── <mesh_id>.pt
```

The directory name should change if any field affecting compatibility changes, such as the PointNeXt checkpoint, number of anchors, feature layer, or coordinate normalization policy.

## Feature-layer policy

Prefer the final decoder per-point feature immediately before the segmentation head. This layer should already be upsampled back to the 4096 input anchors and should provide richer geometry and part-level semantics than final logits.

Implementation options:

1. Modify the PointNeXt segmentation model forward path to optionally return `(logits, pre_logits_feature)`.
2. Register a forward hook on the module immediately before the segmentation classifier head.
3. If the exact hook point is uncertain, temporarily save logits in a separate sanity-check key such as `debug_logits`, but do not treat logits as the production `surface_feat` unless explicitly running a fallback experiment.

Before saving, enforce `surface_feat.shape[0] == surface_xyz.shape[0] == 4096` and convert the feature layout to `[4096, D]` on CPU float32.

## Downstream usage contract

Candidate snapping must follow this contract:

```python
data = torch.load(feature_pt_path, map_location="cpu")
surface_xyz = data["surface_xyz"]      # [4096, 3]
surface_feat = data["surface_feat"]    # [4096, D]

candidate_xyz = candidate_xyz.float()   # [N, 3], same coordinate frame as surface_xyz
surface_xyz = surface_xyz.float()

dist = torch.cdist(candidate_xyz, surface_xyz)
surface_nn_idx = dist.argmin(dim=1)
surface_dist = dist[torch.arange(candidate_xyz.shape[0]), surface_nn_idx]

snapped_xyz = surface_xyz[surface_nn_idx]
candidate_feat = surface_feat[surface_nn_idx]
```

Downstream clustering or reranking may use `snapped_xyz`, `candidate_feat`, `surface_dist`, and original ZeroKey candidate weights. It should not use PointNeXt features as features of raw off-surface candidates without the nearest-anchor assignment step.

## Storage estimate

For 4096 anchors and float32 tensors:

- `surface_xyz`: `4096 * 3 * 4` bytes, about 48 KiB per mesh.
- `surface_normal`: `4096 * 3 * 4` bytes, about 48 KiB per mesh.
- `surface_feat`: `4096 * D * 4` bytes. For example, `D=64` is about 1.0 MiB per mesh; `D=128` is about 2.0 MiB per mesh; `D=256` is about 4.0 MiB per mesh.

For 3145 meshes, the expected footprint is roughly:

- `D=64`: about 3.4 GiB plus metadata overhead.
- `D=128`: about 6.5 GiB plus metadata overhead.
- `D=256`: about 12.8 GiB plus metadata overhead.

This is well below a 200 GB budget without dense point clouds. Saving dense 20000-point pools or per-view intermediates in production artifacts should be avoided.

## Validation checklist

Each produced `.pt` file should pass these checks before being used by aggregation experiments:

- Required keys exist: `surface_xyz`, `surface_feat`, `mesh_id`, `category`, `sampling_info`, `model_info`, and `norm_info`.
- `surface_xyz.dtype == torch.float32` and `surface_feat.dtype == torch.float32`.
- `surface_xyz.shape == (4096, 3)`.
- `surface_feat.ndim == 2` and `surface_feat.shape[0] == 4096`.
- If present, `surface_normal.shape == (4096, 3)`.
- All tensors are finite.
- `model_info["feature_layer"] == "decoder_final_before_seg_head"` for production artifacts.
- `norm_info["coordinate_system"] == "same_as_zerokey_backproject_xyz"`.
