import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy


def build_mlp(
	n_in: int,
	n_out: int,
	hidden_layers: list[int],
	activation_fn: type[nn.Module] = nn.ReLU,
	dropout: float | None = None,
) -> nn.Module:
	layers = []
	prev_units = n_in
	for units in hidden_layers:
		layers.append(nn.Linear(prev_units, units))
		layers.append(activation_fn())
		if dropout:
			layers.append(nn.Dropout(dropout))
		prev_units = units
	layers.append(nn.Linear(prev_units, n_out))
	return nn.Sequential(*layers)


class NexarClassifier(LightningModule):
	def __init__(
		self,
		lr: float = 1e-3,
		hidden_layers: list[int] = [],
		dropout: float | None = None,
	) -> None:
		super().__init__()
		self.save_hyperparameters()

		pretrained_weights = models.ResNet34_Weights.DEFAULT
		self.image_backbone = models.resnet34(weights=pretrained_weights)
		image_backbone_features = self.image_backbone.fc.in_features
		self.image_backbone.fc = nn.Linear(
			in_features=image_backbone_features,
			out_features=image_backbone_features,
		)
		for param in self.image_backbone.parameters():
			param.requires_grad = False
		for param in self.image_backbone.fc.parameters():
			param.requires_grad = True

		self.mask_flow_backbone = models.resnet34(weights=pretrained_weights)
		mask_flow_backbone_features = self.mask_flow_backbone.fc.in_features
		self.mask_flow_backbone.fc = nn.Identity()

		self.classifier = build_mlp(
			n_in=image_backbone_features + mask_flow_backbone_features,
			n_out=1,
			hidden_layers=hidden_layers,
			dropout=dropout,
		)

		self.loss_fn = nn.BCEWithLogitsLoss()
		self.train_accuracy = Accuracy(task="binary")
		self.val_accuracy = Accuracy(task="binary")

	def forward(self, x):
		image, mask_flow = x["frame"], x["mask_flow"]
		img_emb = self.image_backbone(image)
		mf_emb = self.mask_flow_backbone(mask_flow)
		emb = torch.cat([img_emb, mf_emb], dim=1)
		return self.classifier(emb)

	def training_step(self, batch, batch_idx):
		labels = batch["label"]
		pred = self(batch)
		loss = self.loss_fn(pred.squeeze(), labels.float())
		self.log("train_loss", loss, prog_bar=True, on_epoch=True)
		self.train_accuracy(pred.squeeze(), labels)
		self.log("train_acc", self.train_accuracy, prog_bar=True, on_epoch=True)
		return loss

	def validation_step(self, batch, batch_idx):
		labels = batch["label"]
		pred = self(batch)
		loss = self.loss_fn(pred.squeeze(), labels.float())
		self.log("val_loss", loss, prog_bar=True, on_epoch=True)
		self.val_accuracy(pred.squeeze(), labels)
		self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True)
		return loss

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			[
				{"params": self.image_backbone.parameters(), "lr": self.hparams.lr * 0.1},
				{"params": self.mask_flow_backbone.parameters(), "lr": self.hparams.lr * 0.1},
				{"params": self.classifier.parameters(), "lr": self.hparams.lr},
			],
			lr=self.hparams.lr,
		)
		return {"optimizer": optimizer}


