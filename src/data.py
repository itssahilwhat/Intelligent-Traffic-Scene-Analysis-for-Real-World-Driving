import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class NexarDataset(Dataset):
	def __init__(
		self,
		df: pd.DataFrame,
		frame_idx: int | None = None,
		return_label: bool = True,
		transform=None,
	) -> None:
		self.df = df
		self.frame_idx = frame_idx
		self.return_label = return_label
		self.transform = transform

		self.features_path = df["features_path"].values
		self.n_frames = df["n_frames"].values
		if self.return_label:
			self.labels = df["target"].values

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		data = {"idx": idx}

		frame_idx = (
			self.frame_idx
			if self.frame_idx is not None
			else np.random.randint(0, self.n_frames[idx] - 1)
		)
		data["frame_idx"] = frame_idx

		folder = self.features_path[idx]

		if not os.path.isabs(folder):
			cwd = os.getcwd()
			if os.path.basename(cwd) == 'notebooks':
				project_root = os.path.dirname(cwd)
			else:
				project_root = cwd
			if folder.startswith("../data/"):
				folder = os.path.join(project_root, folder.replace("../data/", "data/"))
			elif folder.startswith("data/"):
				folder = os.path.join(project_root, folder)
			else:
				folder = os.path.join(project_root, folder)

		folder = os.path.normpath(folder)

		frame_loaded = False
		original_frame_idx = frame_idx

		for try_idx in range(self.n_frames[idx]):
			frame_idx = (original_frame_idx + try_idx) % self.n_frames[idx]
			frame_path = os.path.join(folder, "frames", f"{str(frame_idx).zfill(2)}.pt")
			flow_path = os.path.join(folder, "flows", f"{str(frame_idx).zfill(2)}.pt")
			mask_path = os.path.join(folder, "masks", f"{str(frame_idx).zfill(2)}.pt")

			try:
				if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
					continue

				frame = torch.load(frame_path)
				flow = torch.load(flow_path)

				if frame is None or flow is None or not isinstance(frame, torch.Tensor) or not isinstance(flow, torch.Tensor):
					continue

				try:
					mask = torch.load(mask_path)
					if mask is None or not isinstance(mask, torch.Tensor):
						mask = torch.zeros((1, *flow.shape[1:]))
				except (FileNotFoundError, EOFError, ValueError):
					mask = torch.zeros((1, *flow.shape[1:]))

				frame_loaded = True
				break

			except (EOFError, RuntimeError, ValueError):
				continue

		if not frame_loaded:
			reference_shape = None
			for check_idx in range(min(len(self.features_path), 10)):
				try:
					check_folder = self.features_path[check_idx]
					if not os.path.isabs(check_folder):
						cwd = os.getcwd()
						project_root = os.path.dirname(cwd) if os.path.basename(cwd) == 'notebooks' else cwd
						if check_folder.startswith("../data/"):
							check_folder = os.path.join(project_root, check_folder.replace("../data/", "data/"))
						elif check_folder.startswith("data/"):
							check_folder = os.path.join(project_root, check_folder)
					ref_frame_path = os.path.join(check_folder, "frames", "00.pt")
					if os.path.exists(ref_frame_path):
						ref_frame = torch.load(ref_frame_path)
						reference_shape = ref_frame.shape
						break
				except:
					continue

			if reference_shape is None:
				reference_shape = torch.Size([3, 426, 426])

			frame = torch.zeros(reference_shape, dtype=torch.float32)
			flow = torch.zeros((2, *reference_shape[1:]), dtype=torch.float32)
			mask = torch.zeros((1, *reference_shape[1:]), dtype=torch.float32)

		frame = frame * (mask > 0).float()
		flow = flow * (mask > 0).float()
		mask_flow = torch.cat([flow, mask], dim=0)

		if self.transform:
			frame = self.apply_transform(frame)
			mask_flow = self.apply_transform(mask_flow)
		data["frame"] = frame.to(torch.float32)
		data["mask_flow"] = mask_flow.to(torch.float32)

		if self.return_label:
			data["label"] = self.labels[idx]

		return data

	def apply_transform(self, image: torch.Tensor) -> torch.Tensor:
		if image.dtype != torch.float32:
			image = image.float()
		if image.max() > 1.0:
			image = image / 255.0
		return self.transform(image)


class NexarDataModule(LightningDataModule):
	def __init__(
		self,
		train_df: pd.DataFrame,
		batch_size: int = 32,
		val_size: float | None = None,
		transform=None,
		test_transform=None,
		num_workers: int = 4,
	) -> None:
		super().__init__()
		self.train_df = train_df
		self.batch_size = batch_size
		self.val_size = val_size
		self.transform = transform
		self.test_transform = test_transform
		self.num_workers = num_workers

	def setup(self, stage=None) -> None:
		if self.val_size is not None:
			train_df, val_df = train_test_split(
				self.train_df, test_size=self.val_size, stratify=self.train_df["target"]
			)
			self.train_dataset = NexarDataset(train_df, transform=self.transform)
			self.val_dataset = NexarDataset(val_df, transform=self.test_transform)
		else:
			self.train_dataset = NexarDataset(self.train_df, transform=self.transform)

	def train_dataloader(self) -> DataLoader:
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=0,
			pin_memory=False,
		)

	def val_dataloader(self) -> DataLoader | None:
		if not hasattr(self, "val_dataset"):
			return None
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size * 2,
			num_workers=0,
			pin_memory=False,
		)


def pad_to_square(image: torch.Tensor):
	_, h, w = image.shape
	max_dim = max(w, h)
	pad_w = (max_dim - w) // 2
	pad_h = (max_dim - h) // 2
	padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)
	return nn.functional.pad(image, padding, mode="constant", value=0)


