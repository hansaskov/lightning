import torch.nn.functional as F
import torchmetrics
from torch import nn, optim, argmax
from lightning.pytorch import LightningModule

class NN(LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)  # Output layer with a single neuron for binary classification
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1_score = torchmetrics.F1Score(task="binary")

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.accuracy(scores.sigmoid(), y),
                "train_f1_score": self.f1_score(scores.sigmoid(), y),
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
                "val_accuracy": self.accuracy(scores.sigmoid(), y),
                "val_f1_score": self.f1_score(scores.sigmoid(), y),
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": self.accuracy(scores.sigmoid(), y),
                "test_f1_score": self.f1_score(scores.sigmoid(), y),
            }
        )
        return loss


    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        y = y.view(-1, 1)  # Ensure that y has the shape [batch_size, 1]
        loss = self.loss_fn(scores, y.float())
        return loss, scores, y


    def predict_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        preds = (scores.sigmoid() > 0.5).int()  # Threshold predictions with sigmoid
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
