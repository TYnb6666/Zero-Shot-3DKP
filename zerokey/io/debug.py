"""Debug evaluation for KeypointNet keypoint detection with per-mesh and per-prompt IoU."""

import sys
from collections import defaultdict
from operator import itemgetter

from zerokey._defaults import DEFAULT_LOG_DIR
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pytorch3d.structures import Pointclouds
from tqdm import tqdm

from kp_utils import KeypointNetDataset, gen_geo_dists
from kp_utils.data.keypoint_labels import INVERSE_CLASS_MAPPING, CLASS_MAPPING
from zerokey.io.kpnet import KPNetEvaluator


class KPNetEvalDebug(KPNetEvaluator):
    """Extended evaluator that computes per-mesh and per-prompt IoU with debug output.

    Saves predicted and ground-truth keypoint point clouds alongside meshes
    for visual inspection, sorted by IoU score.
    """

    @staticmethod
    def eval_det_cls(pred: dict[str, dict[str, Any]], gt: dict[str, list[Any]], geo_dists: dict[str, npt.NDArray[np.float32]], dist_thresh: float = 0.1, confidence_thresh: float = 0.0) -> dict[str, float]:
        """Compute per-mesh detection IoU using geodesic distance thresholding."""
        ret = {}
        for mesh_name in gt.keys():
            gt_kps = np.asarray(gt[mesh_name], dtype=np.int32)[:, 0]
            npos = len(gt_kps)
            if confidence_thresh > 0:
                selection = np.array(pred[mesh_name]["confidence"]) > confidence_thresh
                pred_kps = np.array(pred[mesh_name]["indices"])[selection].astype(np.int32)
            else:
                pred_kps = np.array(pred[mesh_name]["indices"]).astype(np.int32)
            fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
            fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
            ret[mesh_name] = (npos - fn) / np.maximum(npos + fp, np.finfo(np.float64).eps)

        return ret

    @staticmethod
    def eval_det_prompt(pred: dict[str, dict[str, Any]], gt: dict[str, list[Any]], kp_group: dict[str, set[int]], geo_dists: dict[str, npt.NDArray[np.float32]], dist_thresh: float = 0.1, confidence_thresh: float = 0.0) -> defaultdict[str, dict[str, float]]:
        """Compute per-prompt, per-mesh detection IoU grouped by semantic label.

        Returns:
            Nested dict mapping prompt name -> mesh name -> IoU score.
        """
        ret = defaultdict(dict)
        for prompt, sid in kp_group.items():
            sid = np.fromiter(sid, dtype=np.int32, count=len(sid))
            for mesh_name in gt.keys():
                gt_kps = np.asarray(gt[mesh_name], dtype=np.int32)
                gt_mask = np.isin(gt_kps[:, 1], sid)
                if not gt_mask.any():
                    continue
                gt_kps = gt_kps[gt_mask, 0]
                npos = len(gt_kps)
                pred_mask = np.isin(pred[mesh_name]["labels"], sid)
                pred_kps = np.asarray(pred[mesh_name]["indices"], dtype=np.int32)[pred_mask]
                fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
                fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
                ret[prompt][mesh_name] = (npos - fn) / np.maximum(npos + fp, np.finfo(np.float64).eps)

        return ret

    @staticmethod
    def get_kp_from_samples(test_dataset: KeypointNetDataset, cat_name: str, mesh_id: str) -> Any:
        """Look up a specific mesh sample by category name and mesh ID."""
        class_id = INVERSE_CLASS_MAPPING[cat_name]
        for idx, row in enumerate(test_dataset.samples.itertuples()):
            if row.class_id == class_id and row.mesh_id == mesh_id:  # type: ignore[attr-defined]
                return test_dataset[idx]

    def main_loop(self, use_texture: bool = False, offset: int = 0, gaussian_sigma: float = 0.01) -> dict[str, list[tuple[str, float]]]:
        """Run debug evaluation across all categories, saving meshes and keypoints.

        Computes per-mesh and per-prompt IoU, prints results sorted by score,
        and saves the top-ranked mesh/keypoint PLY files for visual debugging.

        Returns:
            Dict mapping category name to a sorted list of (mesh_id, IoU) tuples.
        """
        all_ious = {}
        prompt_idx = 0
        eval_classes = CLASS_MAPPING.copy()
        eval_classes.pop(INVERSE_CLASS_MAPPING['airplane'])
        for cat_name in eval_classes.values():

            test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)
            geo_dists = {}
            pred_all_iou = {}
            gt_all = {}

            kp_group = defaultdict(set)
            for i in tqdm(range(offset, len(test_dataset)), leave=False):
                try:
                    data: Any = test_dataset[i]
                    mesh, keypoints, class_title, mesh_id, pcd = data
                    kp_list = self.io.get_kp_names_from_lable(class_title, mesh_id, keypoints)
                    if not self.io.check_if_complete(kp_list, class_title, mesh_id):
                        continue
                    for k, v in kp_list.items():
                        kp_group[v[prompt_idx]].add(k)

                    if mesh_id not in pred_all_iou:
                        pred_all_iou[mesh_id] = {}
                        pred_all_iou[mesh_id]["indices"] = []
                        pred_all_iou[mesh_id]["labels"] = []
                    if mesh_id not in gt_all:
                        gt_all[mesh_id] = []
                    pcd = pcd.points_packed().numpy().astype(np.float32)

                    geo_dists[mesh_id] = gen_geo_dists(pcd).astype(np.float32)

                    geo_dists_mesh = geo_dists[mesh_id]
                    geo_dists_mesh[np.isinf(geo_dists_mesh)] = geo_dists_mesh[~np.isinf(geo_dists_mesh)].max()
                    _normalized_geo_dists = geo_dists_mesh / np.max(geo_dists_mesh)

                    kps = self.io.load_kps(class_title, mesh_id)
                    predictions_idx, predictions_label = self.find_nearest_idx(pcd, kps)

                    pred_all_iou[mesh_id]["indices"].extend(predictions_idx.tolist())
                    pred_all_iou[mesh_id]["labels"].extend(predictions_label.tolist())

                    for kp in keypoints:
                        gt_all[mesh_id].append((kp["pcd_info"]["point_index"], kp["semantic_id"]))
                except Exception as e:
                    print(f"Error with {e}", file=sys.stderr)

            print(cat_name)
            dist_thresh = 0.1
            iou = self.eval_det_cls(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
            iou_l = sorted(iou.items(), key=itemgetter(1), reverse=True)
            all_ious[cat_name] = iou_l
            s = ""
            logdir = DEFAULT_LOG_DIR / "Debug-M" / cat_name
            logdir.mkdir(parents=True, exist_ok=True)
            for i, (m, x) in enumerate(iou_l):
                s += "{}({})\t".format(m, x)
                mesh, keypoints, class_title, mesh_id, pcd = self.get_kp_from_samples(test_dataset, cat_name, m)
                assert mesh_id == m
                self.io.save_mesh(mesh, logdir / f'{mesh_id}.ply')
                for kp in keypoints:
                    gt_kps = torch.as_tensor([kp['xyz']])
                    self.io.save_pointcloud(Pointclouds(gt_kps[None]), logdir / f'{mesh_id}_{kp['semantic_id']}_gt.ply')
                if i > 10:
                    break

            print('mIoU-{}: {}'.format(dist_thresh, s))
            iou = self.eval_det_prompt(pred_all_iou, gt_all, kp_group, geo_dists, dist_thresh=dist_thresh)
            iou_l = {k: sum(v.values()) / len(v) for k, v in iou.items()}
            logdir = DEFAULT_LOG_DIR / "Debug-P" / cat_name
            logdir.mkdir(parents=True, exist_ok=True)
            for k, v in sorted(iou_l.items(), key=itemgetter(1), reverse=True):
                print('{} ({}): {}'.format(k, kp_group[k], v), end='. ')
                for i, m in enumerate(sorted(iou[k].items(), key=itemgetter(1), reverse=True)):
                    print(m, end=' ')
                    mesh, keypoints, class_title, mesh_id, pcd = self.get_kp_from_samples(test_dataset, cat_name, m[0])
                    assert mesh_id == m[0]
                    self.io.save_mesh(mesh, logdir / f'{mesh_id}.ply')
                    gt_kps = torch.as_tensor([kp['xyz'] for kp in keypoints if kp['semantic_id'] in kp_group[k]])
                    self.io.save_pointcloud(Pointclouds(gt_kps[None]), logdir / f'{mesh_id}_{k}_gt.ply')
                    pred_kps = None
                    for pred_k, pred_v in self.io.load_kps(cat_name, mesh_id).items():
                        if int(pred_k) not in kp_group[k]:
                            continue
                        if pred_kps is None:
                            pred_kps = pred_v
                        else:
                            try:
                                assert torch.allclose(pred_kps, pred_v)
                            except Exception:
                                print(f"pred_kps and pred_v does not match with {mesh_id} {k}", file=sys.stderr)
                    assert pred_kps is not None
                    self.io.save_pointcloud(Pointclouds(pred_kps[None]), logdir / f'{mesh_id}_{k}_pred.ply')
                    if i > 5:
                        break
                print('.')

        return all_ious
