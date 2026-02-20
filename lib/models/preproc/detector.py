from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import scipy.signal as signal

from ultralytics import YOLO

try:
    from mmpose.apis import (
        inference_top_down_pose_model,
        init_pose_model,
        get_track_id,
    )
    _HAS_MMPOSE = True
except Exception:
    _HAS_MMPOSE = False

ROOT_DIR = osp.abspath(f"{__file__}/../../../../")
VIT_DIR = osp.join(ROOT_DIR, "third-party/ViTPose")

VIS_THRESH = 0.3
BBOX_CONF = 0.5
TRACKING_THR = 0.1
MINIMUM_FRMAES = 30
MINIMUM_JOINTS = 6


def _iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    denom = area1 + area2 - inter + 1e-6
    return inter / denom


class DetectionModel(object):
    def __init__(self, device, backend='auto'):
        self.device = str(device).lower()
        self.backend = None
        self.initialize_tracking()

        if backend in ('auto', 'vitpose'):
            self._try_init_vitpose()

        if self.backend is None:
            self._init_yolo_pose()

    def _try_init_vitpose(self):
        pose_model_cfg = osp.join(
            VIT_DIR,
            'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
        )
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'vitpose-h-multi-coco.pth')

        if not _HAS_MMPOSE:
            return
        if not osp.exists(pose_model_cfg):
            return

        try:
            self.pose_model = init_pose_model(pose_model_cfg, pose_model_ckpt, device=self.device)
            bbox_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x.pt')
            if not osp.exists(bbox_model_ckpt):
                bbox_model_ckpt = 'yolov8x.pt'
            self.bbox_model = YOLO(bbox_model_ckpt)
            self.backend = 'vitpose'
        except Exception:
            self.backend = None

    def _init_yolo_pose(self):
        # Works without ViTPose/MMCV and is suitable for macOS CPU/MPS.
        pose_model_ckpt = osp.join(ROOT_DIR, 'checkpoints', 'yolov8x-pose.pt')
        if not osp.exists(pose_model_ckpt):
            pose_model_ckpt = 'yolov8x-pose.pt'
        self.pose_model = YOLO(pose_model_ckpt)
        self.backend = 'yolo_pose'

    def initialize_tracking(self):
        self.next_id = 0
        self.frame_id = 0
        self.pose_results_last = []
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': [],
        }

    def xyxy_to_cxcys(self, bbox, s_factor=1.05):
        cx, cy = bbox[[0, 2]].mean(), bbox[[1, 3]].mean()
        scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200 * s_factor
        return np.array([[cx, cy, scale]])

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        if len(X) == 0:
            self.tracking_results['bbox'] = np.zeros((0, 3))
            return

        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(), kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb

        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox

    def _track_vitpose(self, img, fps):
        bboxes = self.bbox_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        bboxes = [{'bbox': bbox} for bbox in bboxes]

        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            img,
            person_results=bboxes,
            format='xyxy',
            return_heatmap=False,
            outputs=None,
        )

        pose_results, self.next_id = get_track_id(
            pose_results,
            self.pose_results_last,
            self.next_id,
            use_oks=False,
            tracking_thr=TRACKING_THR,
            use_one_euro=True,
            fps=fps,
        )

        for pose_result in pose_results:
            n_valid = (pose_result['keypoints'][:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS:
                continue

            _id = pose_result['track_id']
            xyxy = pose_result['bbox']
            bbox = self.xyxy_to_cxcys(xyxy)

            self.tracking_results['id'].append(_id)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(pose_result['keypoints'])

        self.pose_results_last = pose_results

    def _assign_track_ids(self, detections):
        prev = {x['track_id']: x['bbox'] for x in self.pose_results_last}
        used_prev = set()

        for det in detections:
            best_id, best_iou = None, 0.0
            for track_id, last_bbox in prev.items():
                if track_id in used_prev:
                    continue
                iou = _iou_xyxy(det['bbox'], last_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_id is not None and best_iou >= TRACKING_THR:
                det['track_id'] = best_id
                used_prev.add(best_id)
            else:
                det['track_id'] = self.next_id
                self.next_id += 1

    def _track_yolo_pose(self, img):
        pred = self.pose_model.predict(
            img, device=self.device, classes=0, conf=BBOX_CONF, save=False, verbose=False
        )[0]

        boxes = pred.boxes.xyxy.detach().cpu().numpy() if pred.boxes is not None else np.zeros((0, 4))
        if pred.keypoints is None:
            keypoints = np.zeros((0, 17, 3), dtype=np.float32)
        else:
            keypoints = pred.keypoints.data.detach().cpu().numpy()

        detections = []
        n = min(len(boxes), len(keypoints))
        for i in range(n):
            kp = keypoints[i]
            n_valid = (kp[:, -1] > VIS_THRESH).sum()
            if n_valid < MINIMUM_JOINTS:
                continue
            detections.append({'bbox': boxes[i], 'keypoints': kp})

        self._assign_track_ids(detections)

        self.pose_results_last = [
            {'track_id': det['track_id'], 'bbox': det['bbox']} for det in detections
        ]

        for det in detections:
            bbox = self.xyxy_to_cxcys(det['bbox'])
            self.tracking_results['id'].append(det['track_id'])
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(det['keypoints'])

    def track(self, img, fps, length):
        if self.backend == 'vitpose':
            self._track_vitpose(img, fps)
        else:
            self._track_yolo_pose(img)

        self.frame_id += 1

    def process(self, fps):
        if len(self.tracking_results['id']) == 0:
            return defaultdict(lambda: defaultdict(list))

        for key in ['id', 'frame_id', 'keypoints']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        self.compute_bboxes_from_keypoints()

        output = defaultdict(lambda: defaultdict(list))
        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id':
                    continue
                output[_id][key] = val[idxs]

        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue

            kernel = int(int(fps / 2) / 2) * 2 + 1
            kernel = min(kernel, len(output[_id]['bbox']) if len(output[_id]['bbox']) % 2 == 1 else len(output[_id]['bbox']) - 1)
            if kernel < 3:
                continue
            smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
            output[_id]['bbox'] = smoothed_bbox

        return output
