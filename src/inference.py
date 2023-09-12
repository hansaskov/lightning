from torchvision import datasets, transforms
from model import NN
import config
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

transform = transforms.Compose([
                transforms.Resize([config.IMAGE_HEIGHT, config.IMAGE_WIDTH]),
                transforms.ToTensor(),
                transforms.Normalize(config.DATA_MEAN, config.DATA_STD),
        ])

## Create dataloader
dataset = datasets.ImageFolder('experiment', transform=transform)
# Step 3: Create a DataLoader with pin_memory for data loading optimization
loader = DataLoader(dataset, shuffle=False, num_workers=config.NUM_WORKERS, batch_size=config.BATCH_SIZE, pin_memory=True)

## Load model with checkpoints
model = NN.load_from_checkpoint(
        'logs/lightning_logs/version_6/checkpoints/epoch=2-step=699.ckpt',
        input_size=config.INPUT_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_classes=config.NUM_CLASSES,
    )


trainer = Trainer()
## 
predictions = trainer.predict(model, loader)

print(predictions)