"""Crowd-counting benchmark: YOLO-CROWD vs YOLOv5s vs CSRNet on a YOLO-format split.

Ground truth = number of label lines per image. Reports MAE / RMSE / bias overall and per density bucket.

Usage (from the repository root):
    $ python tools/benchmark_counting.py --split yolo_test/test --json runs/count_eval.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from detect import CSRNet
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics used to train CSRNet
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def yolo_count(model, im0, classes=None, imgsz=640):
    img = letterbox(im0, imgsz, stride=int(model.stride.max()))[0]
    x = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)  # BGR to RGB, 0-1
    t = time.perf_counter()
    det = non_max_suppression(model(x)[0], 0.25, 0.45, classes=classes)[0]  # same thresholds as detect.py
    return len(det), time.perf_counter() - t


def csrnet_count_detectpy(model, im0):
    # replicates detect.py --mode dense: letterbox to 640, RGB / 255, no ImageNet normalisation
    img = letterbox(im0, 640, stride=1)[0]
    x = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0)
    t = time.perf_counter()
    return float(model(x).sum()), time.perf_counter() - t


def csrnet_count_normalized(model, im0):
    # how CSRNet was trained: RGB, ImageNet mean/std, native resolution
    rgb = (im0[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    t = time.perf_counter()
    return float(model(x).sum()), time.perf_counter() - t


def main(opt):
    torch.set_grad_enabled(False)
    device = torch.device('cpu')
    yc = attempt_load(opt.yolo_crowd, map_location=device)
    v5 = attempt_load(opt.yolov5s, map_location=device)
    cs = CSRNet()
    cs.load_state_dict(torch.load(opt.csrnet, map_location=device, weights_only=False))
    cs.eval()

    rows = []
    images = sorted(glob.glob(os.path.join(opt.split, 'images', '*')))
    for i, p in enumerate(images):
        label = os.path.join(opt.split, 'labels', Path(p).stem + '.txt')
        gt = sum(1 for line in open(label) if line.strip()) if os.path.exists(label) else 0
        im0 = cv2.imread(p)
        a, ta = yolo_count(yc, im0)
        b, tb = yolo_count(v5, im0, classes=[0])  # COCO class 0 = person
        c, tc = csrnet_count_detectpy(cs, im0)
        e, te = csrnet_count_normalized(cs, im0)
        rows.append(dict(img=Path(p).name, gt=gt, yolo_crowd=a, yolov5s=b, csrnet_detectpy=c, csrnet_norm=e,
                         t_yolo_crowd=ta, t_yolov5s=tb, t_csrnet_detectpy=tc, t_csrnet_norm=te))
        print(f'{i + 1}/{len(images)} {Path(p).name}: gt={gt} yolo_crowd={a} yolov5s={b} '
              f'csrnet_detectpy={c:.1f} csrnet_norm={e:.1f}')

    if opt.json:
        Path(opt.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(opt.json, 'w'), indent=1)

    gt = np.array([r['gt'] for r in rows], dtype=float)
    keys = ['yolo_crowd', 'yolov5s', 'csrnet_detectpy', 'csrnet_norm']
    print(f'\n{len(rows)} images, GT count mean {gt.mean():.1f} median {np.median(gt):.0f} '
          f'min {gt.min():.0f} max {gt.max():.0f}')
    print(f'{"model":18s}{"MAE":>9s}{"RMSE":>9s}{"bias":>9s}{"median ms":>11s}')
    for k in keys:
        err = np.array([r[k] for r in rows]) - gt
        ms = 1000 * np.median([r['t_' + k] for r in rows])
        print(f'{k:18s}{np.abs(err).mean():9.1f}{np.sqrt((err ** 2).mean()):9.1f}{err.mean():+9.1f}{ms:11.0f}')

    print('\nper density bucket: MAE (mean predicted count)')
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 100), (100, float('inf'))]:
        m = (gt >= lo) & (gt < hi)
        if m.any():
            s = f'GT {lo}-{hi - 1 if hi != float("inf") else "inf"}: n={m.sum()} mean GT={gt[m].mean():.1f} |'
            for k in keys:
                pr = np.array([r[k] for r in rows], dtype=float)[m]
                s += f' {k} {np.abs(pr - gt[m]).mean():.1f} ({pr.mean():.1f}) |'
            print(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='yolo_test/test', help='folder containing images/ and labels/')
    parser.add_argument('--yolo-crowd', default='yolo-crowd.pt', help='YOLO-CROWD weights')
    parser.add_argument('--yolov5s', default='yolov5s.pt', help='COCO YOLOv5s weights')
    parser.add_argument('--csrnet', default='weights.pth', help='CSRNet state_dict')
    parser.add_argument('--json', default='', help='optional path to save per-image results')
    main(parser.parse_args())
