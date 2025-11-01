from __future__ import annotations

import os
import sys
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class ProcessConfig:
	yolo_weights: str = "yolov8m-seg.pt"
	save_every: int = 30
	frames_needed: int = 6
	downsample: int = 3
	vehicle_classes: Tuple[str, ...] = ("car", "truck", "bus", "motorbike", "bicycle")
	device: Optional[str] = None
	max_pos: Optional[int] = None
	max_neg: Optional[int] = None
	max_test: Optional[int] = None


def resolve_device(device_flag: Optional[str]) -> torch.device:
	if device_flag is not None:
		return torch.device(device_flag)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def load_yolo(weights: str, device: torch.device):
	from ultralytics import YOLO
	model = YOLO(weights)
	model.to(device)
	return model


@torch.no_grad()
def extract_mask(frame_rgb: np.ndarray, model, vehicle_classes: Tuple[str, ...]) -> Optional[np.ndarray]:
	model.eval()
	results = model(frame_rgb, verbose=False)[0]
	masks = results.masks
	classes = results.boxes.cls
	if masks is None or masks.data is None or len(masks.data) == 0:
		return None
	H, W = frame_rgb.shape[:2]
	mask_out = np.zeros((H, W), dtype=np.uint8)
	masks_np = masks.data.detach().cpu().numpy().astype(np.uint8)
	classes_np = classes.detach().cpu().numpy()
	names = results.names if hasattr(results, "names") else getattr(model.model, "names", {})
	for seg, cls_id in zip(masks_np, classes_np):
		cls_id = int(cls_id)
		class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
		if class_name in vehicle_classes:
			mask = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
			mask_out[mask > 0] = 255
	return mask_out[None, ...] if mask_out.any() else None


def compute_flow_channels(frame1_rgb: np.ndarray, frame2_rgb: np.ndarray) -> np.ndarray:
	prev_gray = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2GRAY)
	next_gray = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2GRAY)
	flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
	angle = (angle * 180 / np.pi / 2).astype(np.uint8)
	flow_hw2 = np.dstack((magnitude, angle))
	return np.moveaxis(flow_hw2, -1, 0)


def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
	if img.ndim == 3 and img.shape[0] <= 3:
		img = np.transpose(img, (1, 2, 0))
	h, w = img.shape[:2]
	ds = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_LINEAR)
	if ds.ndim == 2:
		ds = ds[..., None]
	return np.transpose(ds, (2, 0, 1))


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def iter_rows(df: pd.DataFrame, cap: Optional[int]):
	it = df.to_dict(orient="records")
	if cap is not None:
		it = it[:cap]
	for r in it:
		yield r


def save_sample(base_dir: str, idx: int, frame_rgb: np.ndarray, flow_ch2: np.ndarray, mask1: Optional[np.ndarray], downsample: int) -> None:
	save_dir = base_dir
	ensure_dir(save_dir)
	flow_path = os.path.join(save_dir, f"flows/{str(idx).zfill(2)}.pt")
	frame_path = os.path.join(save_dir, f"frames/{str(idx).zfill(2)}.pt")
	ensure_dir(os.path.dirname(flow_path)); ensure_dir(os.path.dirname(frame_path))
	flow_ds = downsample_image(flow_ch2, downsample)
	frame_ds = downsample_image(frame_rgb, downsample)
	torch.save(torch.tensor(flow_ds, dtype=torch.float32), flow_path)
	torch.save(torch.tensor(frame_ds, dtype=torch.int16), frame_path)
	if mask1 is not None:
		mask_path = os.path.join(save_dir, f"masks/{str(idx).zfill(2)}.pt")
		ensure_dir(os.path.dirname(mask_path))
		mask_ds = downsample_image(mask1, downsample)
		torch.save(torch.tensor(mask_ds, dtype=torch.int16), mask_path)


def process_negatives(train_df: pd.DataFrame, cfg: ProcessConfig, model, device: torch.device) -> None:
	dq: deque[np.ndarray] = deque(maxlen=3)
	neg_df = train_df[train_df["target"] == 0]
	for row in tqdm(iter_rows(neg_df, cfg.max_neg), total=len(neg_df) if cfg.max_neg is None else min(cfg.max_neg, len(neg_df)), desc="process_negatives"):
		vid_path = str(RAW_DIR / "train" / f"{int(row['id']):05d}.mp4")
		cap = cv2.VideoCapture(vid_path)
		if not cap.isOpened():
			continue
		frame_idx, save_idx = 0, 0
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				break
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			if frame_idx % cfg.save_every == 0 and len(dq) == 3:
				mask1 = extract_mask(frame_rgb, model, cfg.vehicle_classes)
				flow_ch2 = compute_flow_channels(dq[0], frame_rgb)
				base = str(PROC_DIR / "train" / f"{int(row['id']):05d}")
				save_sample(base, save_idx, frame_rgb, flow_ch2, mask1, cfg.downsample)
				save_idx += 1
			frame_idx += 1
			dq.append(frame_rgb)
		cap.release()


def process_positives(train_df: pd.DataFrame, cfg: ProcessConfig, model, device: torch.device) -> None:
	dq: deque[np.ndarray] = deque(maxlen=3)
	pos_df = train_df[(train_df["target"] == 1) & train_df["time_of_alert"].notna() & train_df["time_of_event"].notna()]
	fps_default = 30.0
	for row in tqdm(iter_rows(pos_df, cfg.max_pos), total=len(pos_df) if cfg.max_pos is None else min(cfg.max_pos, len(pos_df)), desc="process_positives"):
		vid_path = str(RAW_DIR / "train" / f"{int(row['id']):05d}.mp4")
		cap = cv2.VideoCapture(vid_path)
		if not cap.isOpened():
			continue
		fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
		t_alert = float(row["time_of_alert"]) ; t_event = float(row["time_of_event"]) ;
		start_f = int(math.floor(t_alert * fps)) ; end_f = int(math.ceil(t_event * fps))
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
		f_idx, save_idx = start_f, 0
		while f_idx <= end_f:
			ok, frame_bgr = cap.read()
			if not ok:
				break
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			dq.append(frame_rgb)
			if len(dq) == 3:
				mask1 = extract_mask(frame_rgb, model, cfg.vehicle_classes)
				flow_ch2 = compute_flow_channels(dq[0], frame_rgb)
				base = str(PROC_DIR / "train" / f"{int(row['id']):05d}")
				save_sample(base, save_idx, frame_rgb, flow_ch2, mask1, cfg.downsample)
				save_idx += 1
			f_idx += 1
		cap.release()


def process_test(test_df: pd.DataFrame, cfg: ProcessConfig, model, device: torch.device) -> None:
	dq: deque[np.ndarray] = deque(maxlen=3)
	for row in tqdm(iter_rows(test_df, cfg.max_test), total=len(test_df) if cfg.max_test is None else min(cfg.max_test, len(test_df)), desc="process_test"):
		vid_path = str(RAW_DIR / "test" / f"{int(row['id']):05d}.mp4")
		cap = cv2.VideoCapture(vid_path)
		if not cap.isOpened():
			continue
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		start_f = max(0, total - cfg.frames_needed)
		cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
		f_idx, save_idx = start_f, 0
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				break
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			dq.append(frame_rgb)
			if f_idx >= total - 3 and len(dq) == 3:
				mask1 = extract_mask(frame_rgb, model, cfg.vehicle_classes)
				flow_ch2 = compute_flow_channels(dq[0], frame_rgb)
				base = str(PROC_DIR / "test" / f"{int(row['id']):05d}")
				save_sample(base, save_idx, frame_rgb, flow_ch2, mask1, cfg.downsample)
				save_idx += 1
			f_idx += 1
		cap.release()


def finalize_metadata() -> None:
	train_csv = RAW_DIR / "train.csv"
	test_csv = RAW_DIR / "test.csv"
	train_df = pd.read_csv(train_csv)
	test_df = pd.read_csv(test_csv)
	def count_frames(base_dir: str) -> int:
		frames_dir = os.path.join(base_dir, "frames")
		try:
			return len([f for f in os.listdir(frames_dir) if f.endswith(".pt")])
		except Exception:
			return 0
	train_df["features_path"] = train_df["id"].apply(lambda x: os.path.join("../data/processed/train", f"{int(x):05d}"))
	test_df["features_path"] = test_df["id"].apply(lambda x: os.path.join("../data/processed/test", f"{int(x):05d}"))
	train_df["n_frames"] = train_df["features_path"].apply(lambda p: count_frames(os.path.normpath(os.path.join(PROJECT_ROOT, p.replace('..', '.')))))
	test_df["n_frames"] = test_df["features_path"].apply(lambda p: count_frames(os.path.normpath(os.path.join(PROJECT_ROOT, p.replace('..', '.')))))
	out_train = PROC_DIR / "train.parquet"
	out_test = PROC_DIR / "test.parquet"
	train_df.to_parquet(out_train, index=False)
	test_df.to_parquet(out_test, index=False)
	print("Saved:", out_train)
	print("Saved:", out_test)


def main(cfg: ProcessConfig) -> None:
	PROC_DIR.mkdir(parents=True, exist_ok=True)
	device = resolve_device(cfg.device)
	print("Using device:", device)
	train_csv = RAW_DIR / "train.csv"
	test_csv = RAW_DIR / "test.csv"
	assert train_csv.exists() and test_csv.exists(), "Missing train.csv or test.csv under data/raw"
	train_df = pd.read_csv(train_csv)
	test_df = pd.read_csv(test_csv)
	print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
	yolo = load_yolo(cfg.yolo_weights, device)
	process_negatives(train_df, cfg, yolo, device)
	process_positives(train_df, cfg, yolo, device)
	process_test(test_df, cfg, yolo, device)
	finalize_metadata()
	print("Done.")


def parse_args(argv: Optional[list[str]] = None) -> ProcessConfig:
	import argparse
	p = argparse.ArgumentParser(description="Process raw Nexar data into tensors and parquet")
	p.add_argument("--yolo-weights", type=str, default="yolov8m-seg.pt")
	p.add_argument("--save-every", type=int, default=30)
	p.add_argument("--frames-needed", type=int, default=6)
	p.add_argument("--downsample", type=int, default=3)
	p.add_argument("--device", type=str, default=None)
	p.add_argument("--max-pos", type=int, default=None)
	p.add_argument("--max-neg", type=int, default=None)
	p.add_argument("--max-test", type=int, default=None)
	args = p.parse_args(argv)
	return ProcessConfig(
		yolo_weights=args.yolo_weights,
		save_every=args.save_every,
		frames_needed=args.frames_needed,
		downsample=args.downsample,
		device=args.device,
		max_pos=args.max_pos,
		max_neg=args.max_neg,
		max_test=args.max_test,
	)


if __name__ == "__main__":
	cfg = parse_args()
	main(cfg)


