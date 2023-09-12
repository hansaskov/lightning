import torch
import torch.nn.functional as F
import torchmetrics
from torchvision.models import resnet18
from torch import argmax, nn, optim, FloatTensor
from lightning.pytorch import LightningModule


class NN(LightningModule):
    def __init__(self, input_size, num_classes, learning_rate, class_weighting):
        super().__init__()
        self.lr = learning_rate
        backbone = resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=FloatTensor(class_weighting))  # Cross-entropy loss
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.accuracy(scores, y),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "scores": scores, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": self.accuracy(scores, y),
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": self.accuracy(scores.sigmoid(), y),
            }
        )
        return loss


    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y


    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    