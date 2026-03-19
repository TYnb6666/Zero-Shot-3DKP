"""I/O and evaluation utilities for KeypointNet-based 3D keypoint detection."""

import itertools
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from typing import Any, Generator, Iterable, Iterator, Union

from einops import rearrange
import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.io import IO
from pytorch3d.structures import Meshes, Pointclouds, join_pointclouds_as_scene
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from kp_utils import KeypointNetDataset, eval_iou, gen_geo_dists, CLASS_MAPPING


def setids_to_uint8_flags(setids: Iterable[int]) -> torch.Tensor:
    """Pack semantic IDs (0-23) into a 3-byte uint8 bitmask tensor.

    Each ID sets one bit in a 24-bit field, stored as 3 big-endian uint8 values.
    Used to encode which semantic keypoints are present in a single filename-safe
    hex string (e.g. ``0x{:02X}{:02X}{:02X}``).

    Args:
        setids: Semantic IDs, each an integer in [0, 23].

    Returns:
        Tensor of shape [3] with dtype uint8, big-endian bit encoding.

    Raises:
        ValueError: If any ID is not an integer in [0, 23].
    """
    # Build a 24-bit bitmask where bit i is set iff semantic ID i is present.
    # Example: IDs {0, 3, 7} -> flags = 0b10001001 = 0x000089
    flags = 0
    for id in setids:
        if not isinstance(id, int) or id >= 24 or id < 0:
            raise ValueError(f"ID {id} must be an integer between 0 and 23")
        flags |= (1 << id)  # Set the bit at position id
    # Convert to 3 big-endian bytes and wrap as uint8 tensor for PLY feature storage.
    # Big-endian ensures the hex representation is human-readable in filenames.
    return torch.frombuffer(flags.to_bytes(length=3, byteorder='big'), dtype=torch.uint8)

class KPNetIO(IO):
    """Handles file I/O for KeypointNet keypoint detection experiments.

    Manages saving/loading of meshes, point clouds, keypoint files, and rendered
    image grids. Organizes output under ``output_dir/class_title/mesh_id/``.

    Extends PyTorch3D's ``IO`` for mesh and point cloud serialization.

    Attributes:
        output_dir: Root directory for all experiment outputs.
        jdict: Mapping from class name to per-mesh semantic ID annotations,
            loaded from ``semantic_names.json``.
    """

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.output_dir = Path(output_dir)
        self.jdict = self.load_semantic_names()
        super(KPNetIO, self).__init__()

    @staticmethod
    def load_semantic_names() -> defaultdict[str, dict[str, Any]]:
        """Load semantic keypoint name annotations from ``semantic_names.json``.

        Returns:
            Nested dict: ``{class_name: {model_id: {semantic_id: name_str}}}``.
        """
        jdict: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        with Path('data_creation', 'keypointnet', 'semantic_names.json').open('r') as f:
            j = json.load(f)
        for item in j:
            jdict[CLASS_MAPPING[item['class_id']]][item['model_id']] = item['semantic_id_mapping']
        return jdict

    def get_kp_names_from_lable(self, class_title: str, mesh_id: str, keypoints: Any) -> dict[Any, Any]:
        """Build prompt lists for each semantic keypoint ID.

        Combines the ground-truth label from the dataset with any additional
        semantic name annotations found in ``self.jdict``.

        Returns:
            Dict mapping ``semantic_id -> [label, *annotation_names]``.
        """
        jdict = self.jdict.get(class_title)
        ret_dict: defaultdict[int, list[str]] = defaultdict(list)

        for kps in keypoints:
            semantic_id = kps['semantic_id']
            ret_dict[semantic_id].append(kps['label'][0])
            for anno_dict in jdict.values() if jdict is not None else ():
                if anno_str := anno_dict.get(str(semantic_id), None):
                    if anno_str == 'None':
                        continue
                    ret_dict[semantic_id].append(anno_str)
        return ret_dict

    @staticmethod
    def loop_over_test_datasets(use_texture: bool) -> Iterator[tuple[Meshes, Any, str, str, Pointclouds]]:
        """Yield test meshes from KeypointNet, interleaving across categories.

        Iterates over airplane, chair, and table datasets in round-robin order
        using ``zip_longest``, yielding ``(mesh, keypoints, class_title, mesh_id, pcd)``.
        """
        test_datasets: Any = (
            KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture) for cat_name in
            ['airplane', 'chair', 'table']  # CLASS_MAPPING.values()
        )
        for batch in zip_longest(*test_datasets):
            yield from filter(None, batch)

    def check_if_complete(self, kp_list: dict[Any, Any], class_title: str, mesh_id: str) -> bool:
        """Check whether keypoint detection results already exist for this mesh.

        Falls back to ``check_if_complete2`` for the legacy flat directory layout.
        """
        save_dir = self.output_dir / class_title / mesh_id
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return self.check_if_complete2(kp_list, class_title, mesh_id)
        # Encode all semantic IDs as a 3-byte hex bitmask for the filename.
        # E.g. IDs {0, 1, 5} -> bitmask 0x000023 -> filename "mesh_0x000023_keypts.ply"
        all_semantic_ids = setids_to_uint8_flags(kp_list.keys())
        semantic_id='0x{:02X}{:02X}{:02X}'.format(*all_semantic_ids.tolist())
        return mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists()

    def check_if_complete2(self, kp_list: dict[Any, Any], class_title: str, mesh_id: str) -> bool:
        """Legacy completeness check using flat ``class_title/`` directory layout."""
        save_dir = self.output_dir / class_title
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return False
        for semantic_id, _kp in kp_list.items():
            if not mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists():
                return False
        return True

    def save_kps(self, mesh: Meshes, kps_3d: Pointclouds, class_title: str, mesh_id: str, semantic_id: str, postfix: str = 'keypts', prefix: str | None = None) -> None:
        """Save detected keypoints and associated mesh to PLY files.

        Writes the mesh (if not already saved) and the keypoint point cloud.
        Filename format: ``{prefix}_{semantic_id}_{postfix}.ply``.
        """
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        kps_3d = join_pointclouds_as_scene(kps_3d)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{prefix or mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title: str, mesh_id: str, postfix: str = 'keypts', prefix: str | None = None) -> dict[Any, Any]:
        """Load saved keypoints, keyed by packed semantic bitmask (int).

        Falls back to ``load_kps2`` for the legacy flat directory layout.

        Returns:
            Dict mapping bitmask int -> points tensor [N, 3].
        """
        save_dir = self.output_dir / class_title / mesh_id
        if not save_dir.exists():
            return self.load_kps2(class_title, mesh_id, postfix)
        # Decode hex bitmask from filename back to integer key.
        # Filename: "{mesh_id}_0x{HH}{HH}{HH}_keypts.ply" -> split on '_', take [1],
        # strip '0x' prefix, decode hex bytes, interpret as big-endian integer.
        # The integer is the same 24-bit bitmask produced by setids_to_uint8_flags.
        all_kps = {int.from_bytes(bytes.fromhex(fname.stem.split('_')[1].removeprefix('0x')), 'big'):
                   self.load_pointcloud(fname).points_packed()
                   for fname in save_dir.glob(f"{prefix or mesh_id}_0x*_{postfix}.ply")}
        return all_kps

    def load_kps2(self, class_title: str, mesh_id: str, postfix: str = 'keypts') -> dict[Any, Any]:
        """Legacy keypoint loader using flat ``class_title/`` directory layout."""
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed()
               for fname in save_dir.glob(f"{mesh_id}_*_{postfix}.ply")}
        return kps

    def load_kps_with_semantic_ids(self, class_title: str, mesh_id: str, postfix: str = 'keypts', prefix: str | None = None) -> dict[frozenset[int], torch.Tensor]:
        """Load keypoints keyed by frozenset of semantic IDs.

        Returns:
            Dict mapping ``frozenset({semantic_id, ...})`` -> points tensor [N, 3].
        """
        save_dir = self.output_dir / class_title / mesh_id
        all_kps = {frozenset(map(int, fname.stem.split('_')[1].split(','))):
                   self.load_pointcloud(fname).points_packed()
                   for fname in save_dir.glob(f"{prefix or mesh_id}_*_{postfix}.ply")}
        return all_kps

    def save_kps_with_semantic_ids(self, mesh: Meshes, all_kps: dict[frozenset[Any], Pointclouds], class_title: str, mesh_id: str, postfix: str = 'keypts', prefix: str | None = None) -> None:
        """Save keypoints with semantic ID bitmask features.

        Concatenates all keypoint groups into a single point cloud, attaching
        a 3-byte uint8 bitmask feature indicating which semantic IDs each
        keypoint corresponds to. The combined bitmask of all IDs is encoded
        in the filename.
        """
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)

        all_kps_3d = []
        all_flags = []
        for semantic_ids, kps in all_kps.items():
            pts = kps.points_packed()
            all_kps_3d.append(pts.cpu())
            # Create feature tensor same shape as points
            flags = setids_to_uint8_flags(semantic_ids)
            all_flags.append(flags.expand(pts.shape[0], -1))

        all_semantic_ids = setids_to_uint8_flags(itertools.chain.from_iterable(all_kps.keys()))
        kps_3d = Pointclouds(points=all_kps_3d, features=all_flags)
        self.save_kps(mesh, kps_3d, class_title, mesh_id,
                      semantic_id='0x{:02X}{:02X}{:02X}'.format(*all_semantic_ids.tolist()),
                      postfix=postfix, prefix=prefix)

    def save_images(self, images: Union[torch.Tensor, list[Image.Image]], class_title: str, mesh_id: str, postfix: str = 'views', prefix: str | None = None) -> None:
        """Save rendered views as a grid image (PNG).

        Accepts either a batch tensor [B, C, H, W] or a list of PIL Images.
        Arranges them into a roughly square grid and saves at 200 DPI.
        """
        save_dir = self.output_dir / class_title / mesh_id
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(images, torch.Tensor):
            frames = rearrange(images, 'b c h w -> b h w c').cpu()
            images = [Image.fromarray(frame.numpy()) for frame in frames]

        num_images = len(images)
        rows = int(np.ceil(np.sqrt(num_images)))
        cols = int(np.ceil(num_images / rows))

        plt.ioff()
        plt.figure(figsize=(4*cols, 4*rows))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix or mesh_id}_{postfix}.png", bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()


class KPNetEvaluator:
    """Evaluates detected keypoints against KeypointNet ground truth using geodesic IoU.

    Loads predicted keypoints from disk, snaps them to the nearest point cloud
    vertex, computes geodesic distances, and reports mIoU at multiple thresholds.

    Attributes:
        io: Primary I/O handler for loading predictions.
        ioref: Optional reference I/O handlers for cross-experiment filtering.
    """

    def __init__(self, log_dir: str | os.PathLike[str] = Path(), expname: str = 'BackprojectPTS', ioref: tuple[str, ...] = ()) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.io = KPNetIO(Path(log_dir, expname))
        self.ioref = [KPNetIO(Path(log_dir, refname)) for refname in ioref]

    @staticmethod
    def find_nearest_idx(pcd: npt.NDArray[np.floating[Any]], kps: dict[Any, torch.Tensor]) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.signedinteger[Any]]]:
        """Snap predicted keypoints to the nearest point cloud vertex.

        Args:
            pcd: Point cloud vertices [V, 3].
            kps: Dict mapping label key -> predicted keypoint positions [N, 3].

        Returns:
            Tuple of (vertex_indices, labels) where vertex_indices[i] is the
            nearest vertex to the i-th predicted keypoint, and labels[i] is
            its corresponding semantic label.
        """
        # Unzip {label: points} dict into parallel lists: one of point tensors,
        # one of label tensors (each label repeated to match its points count).
        # Then concatenate both and use NearestNeighbors to map each point to
        # its closest vertex in the point cloud.
        kps_tensors, label_tensors = zip(*((v, torch.full(v.shape[:1], fill_value=int(k), dtype=torch.int)) for k, v in kps.items()))
        kps_np = torch.cat(kps_tensors, dim=0).cpu().numpy()
        label = torch.cat(label_tensors, dim=0).cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=1).fit(pcd)
        indices = nbrs.kneighbors(kps_np, return_distance=False)
        return indices.squeeze(), label

    def main_loop(self, use_texture: bool = False, offset: int = 0, gaussian_sigma: float = 0.01) -> None:
        """Run IoU evaluation across all KeypointNet categories.

        For each mesh, loads predictions, snaps to vertices, computes pairwise
        geodesic distances, and accumulates per-category IoU at thresholds
        from 0.00 to 0.20 in 0.01 increments.

        Args:
            use_texture: Whether to load textured meshes.
            offset: Starting index within each category's dataset.
            gaussian_sigma: Unused (reserved for Gaussian heatmap evaluation).
        """
        for cat_name in ['table', 'airplane', 'chair']:  # CLASS_MAPPING.values()

            test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)
            geo_dists = {}

            pred_all_iou = {
                cat_name: {}
            }
            gt_all = {
                cat_name: {}
            }

            pbar = tqdm(range(offset, len(test_dataset)), leave=False)
            for i in pbar:
                data: Any = test_dataset[i]
                _mesh, keypoints, class_title, mesh_id, pcd = data
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                if not self.io.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                for ioref in self.ioref:
                    if not ioref.check_if_complete(kp_list, class_title, mesh_id):
                        continue

                if mesh_id not in pred_all_iou[cat_name]:
                    pred_all_iou[cat_name][mesh_id] = {}
                    pred_all_iou[cat_name][mesh_id]["indices"] = []
                    pred_all_iou[cat_name][mesh_id]["confidence"] = []
                if mesh_id not in gt_all[cat_name]:
                    gt_all[cat_name][mesh_id] = []
                pcd = pcd.points_packed().numpy().astype(np.float32)

                geo_dists[mesh_id] = gen_geo_dists(pcd).astype(np.float32)

                geo_dists_mesh = geo_dists[mesh_id]
                geo_dists_mesh[np.isinf(geo_dists_mesh)] = geo_dists_mesh[~np.isinf(geo_dists_mesh)].max()
                _normalized_geo_dists = geo_dists_mesh / np.max(geo_dists_mesh)

                kps = self.io.load_kps(class_title, mesh_id)
                predictions_idx, _ = self.find_nearest_idx(pcd, kps)
                # predictions_idx, selection_matrix = optimize_keypoint_candidates(kp_dists, normalized_geo_dists,
                #                                                                  kp_features, features,
                #                                                                  num_steps=5000, lr=0.1, device=device,
                #                                                                  dist_alpha=dist_weight,
                #                                                                  selection_beta=0)

                # pred_all_iou[cat_name][mesh_id]["confidence"].extend(selection_matrix[:, :-1].max(axis=0).tolist())
                pred_all_iou[cat_name][mesh_id]["indices"].extend(predictions_idx.tolist())

                for kp in keypoints:
                    gt_all[cat_name][mesh_id].append(kp["pcd_info"]["point_index"])

                pbar.set_description(f"Processing {cat_name} mIoU-0.1: %({cat_name})s" % eval_iou(
                    pred_all_iou, gt_all, geo_dists, dist_thresh=0.1))

            print(cat_name)
            for i in range(21):
                dist_thresh = 0.01 * i
                iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
                iou_l = list(iou.values())
                s = ""
                for x in iou_l:
                    s += "{}\t".format(x)
                print('mIoU-{}: {}'.format(dist_thresh, s))


class RefIO(KPNetIO):
    """Base I/O handler for reference-view feature accumulation.

    Collects patch features from reference views by matching ground-truth
    keypoint locations to nearest patch centers, then provides averaged
    reference features for subsequent matching. Shared by PatchAlign3DRefIO
    and ULIP2RefIO.

    Attributes:
        reference_global: Accumulated patch features per (class, semantic_id).
        used_as_reference: Count of reference views consumed per class.
        max_reference_features: Maximum reference views to average over.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reference_global: defaultdict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
        self.used_as_reference: Counter[str] = Counter()
        self.max_reference_features: int = 3

    def get_kp_names_from_lable(self, class_title: str, mesh_id: str, keypoints: Any) -> dict[Any, Any]:
        """Return ground-truth keypoint 3D locations keyed by semantic_id."""
        return {int(k['semantic_id']): torch.atleast_2d(torch.as_tensor(k['xyz'])) for k in keypoints}

    @contextmanager
    def sample_reference_view(self, npz_file: bytes, keypoints: dict[int, torch.Tensor], mesh: Meshes, class_title: str) -> Generator[bool, None, None]:
        """Context manager that accumulates reference features or signals readiness.

        Yields True when enough reference views have been collected. If the caller
        raises LookupError, intercepts it to extract and store patch features
        from the npz data, then raises IOError to skip this mesh instance.

        The exception-based control flow works as follows:
          1. Yields ``used_as_reference >= max_reference_features`` (True/False).
          2. If the caller sees False (not enough references), it raises LookupError.
          3. The except block catches LookupError, extracts patch features from
             npz_file, finds the nearest patches to each ground-truth keypoint,
             stores them in reference_global, and raises IOError to abort this mesh.
          4. If the caller sees True, normal execution continues (no exception).
        """
        from zerokey.generators.patchalign3d import transform_to_mesh_space

        try:
            yield self.used_as_reference[class_title] >= self.max_reference_features
        except LookupError:
            with BytesIO(npz_file) as in_buffer:
                kp = np.load(in_buffer, allow_pickle=True)
                device = mesh.verts_packed().device  # type: ignore[union-attr]
                patch_centers = transform_to_mesh_space(kp, mesh, device, centers_key='patch_centers')
                patch_feat_data = kp.get('patch_feat', kp.get('patch_emb', None))
                if patch_feat_data is None:
                    raise ValueError("Neither patch_feat nor patch_emb found in npz file")
                patch_features = torch.as_tensor(patch_feat_data, device=device)

            indices, labels = KPNetEvaluator.find_nearest_idx(patch_centers.cpu().numpy(), keypoints)
            features = patch_features[indices]
            for idx, feature in zip(labels, features):
                self.reference_global[(class_title, idx)].append(feature)
            self.used_as_reference.update([class_title])
            raise IOError("No reference view found")

    def get_reference_features(self, class_title: str, idx: int) -> torch.Tensor:
        """Return averaged reference patch features for a given class and semantic ID."""
        ret = self.reference_global[(class_title, idx)]
        if not ret:
            raise LookupError("No reference features found")
        return torch.stack(ret[:self.max_reference_features]).mean(dim=0)
