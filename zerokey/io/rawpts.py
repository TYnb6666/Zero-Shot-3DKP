"""Raw-points I/O and evaluation for KeypointNet keypoint detection."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.structures import Meshes, Pointclouds
from tqdm import tqdm

from kp_utils import KeypointNetDataset, eval_iou, gen_geo_dists
from zerokey.io.kpnet import KPNetIO as _KPNetIO, KPNetEvaluator as _KPNetEvaluator


class KPNetIO(_KPNetIO):
    """I/O handler for raw keypoint point clouds on the KeypointNet dataset.

    Unlike the main ``KPNetIO`` in ``zerokey.io.kpnet``, this variant loads
    raw point files (postfix ``'rawpts'``) and averages them to produce a
    single keypoint coordinate per semantic ID.

    Inherits ``__init__``, ``load_semantic_names``, and ``loop_over_test_datasets``
    from the base ``KPNetIO``.
    """

    def get_kp_names_from_lable(self, class_title: str, mesh_id: str, keypoints: Any) -> defaultdict[int, list[str]]:
        """Map semantic IDs to human-readable keypoint names from annotations."""
        jdict = self.jdict.get(class_title)
        ret_dict: defaultdict[int, list[str]] = defaultdict(list)

        for kps in keypoints:
            semantic_id = kps['semantic_id']
            ret_dict[semantic_id].append(kps['label'][0])
            if jdict is None:
                continue
            for anno_dict in jdict.values():
                if anno_str := anno_dict.get(str(semantic_id), None):
                    if anno_str == 'None':
                        continue
                    ret_dict[semantic_id].append(anno_str)
        return ret_dict

    def check_if_complete(self, kp_list: dict[Any, Any], class_title: str, mesh_id: str) -> bool:
        """Check whether all expected keypoint files exist for a given mesh."""
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            return False
        for semantic_id, _kp in kp_list.items():
            if not mesh_file.with_stem(f"{mesh_id}_{semantic_id}_keypts").exists():
                return False
        return True

    def save_kps(self, mesh: Meshes, kps_3d: Pointclouds, class_title: str, mesh_id: str, semantic_id: str, postfix: str = 'keypts') -> None:
        """Save a keypoint point cloud alongside its mesh as PLY files."""
        save_dir = self.output_dir / class_title
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = save_dir / f"{mesh_id}_mesh.ply"
        if not mesh_file.exists():
            self.save_mesh(mesh, mesh_file)
        self.save_pointcloud(kps_3d, mesh_file.with_stem(f"{mesh_id}_{semantic_id}_{postfix}"))

    def load_kps(self, class_title: str, mesh_id: str, postfix: str = 'rawpts') -> dict[Any, Any]:
        """Load raw keypoint files and return their mean coordinates per semantic ID."""
        save_dir = self.output_dir / class_title
        kps = {fname.stem.split('_')[1]: self.load_pointcloud(fname).points_packed().mean(dim=0, keepdim=True)
               for fname in save_dir.glob(f"{mesh_id}_*_{postfix}.ply")}
        return kps


class KPNetEvaluator(_KPNetEvaluator):
    """Evaluates raw keypoint predictions against KeypointNet ground truth.

    Loads predictions from the primary I/O directory and cross-references with
    two reference directories (GPT-4o and GlobalPTS) to ensure completeness
    before computing IoU at multiple geodesic distance thresholds.

    Inherits ``find_nearest_idx`` from the base ``KPNetEvaluator``.
    """

    def __init__(self, log_dir: str | os.PathLike[str] = Path(), expname: str = 'BackprojectPTS') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.io = KPNetIO(Path(log_dir, expname))
        self.ioref1 = KPNetIO(Path(log_dir, 'GPT-4o'))
        self.ioref2 = KPNetIO(Path(log_dir, 'GlobalPTS'))

    def main_loop(self, use_texture: bool = False, offset: int = 0, gaussian_sigma: float = 0.01) -> None:
        """Evaluate raw keypoint predictions and print IoU at 21 distance thresholds."""
        for cat_name in ['airplane', 'chair', 'table']:  # CLASS_MAPPING.values()

            test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)
            geo_dists = {}

            pred_all_iou = {
                cat_name: {}
            }
            gt_all = {
                cat_name: {}
            }

            for i in tqdm(range(offset, len(test_dataset)), leave=False):
                data: Any = test_dataset[i]
                _mesh, keypoints, class_title, mesh_id, pcd = data
                kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                if not self.io.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                if not self.ioref1.check_if_complete(kp_list, class_title, mesh_id):
                    continue
                if not self.ioref2.check_if_complete(kp_list, class_title, mesh_id):
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

            print(cat_name)
            for i in range(21):
                dist_thresh = 0.01 * i
                iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
                iou_l = list(iou.values())
                s = ""
                for x in iou_l:
                    s += "{}\t".format(x)
                print('mIoU-{}: {}'.format(dist_thresh, s))
