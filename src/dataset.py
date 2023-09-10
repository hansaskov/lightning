from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from lightning.pytorch import LightningDataModule

class kickBallDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, train_val_test_split, img_height, img_width):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.img_height = img_height
        self.img_width = img_width

    def setup(self, stage):

        # Load dataset and transform by resizing
        dataset = datasets.ImageFolder(
            self.data_dir, 
            transform = transforms.Compose([
                transforms.Resize([self.img_height, self.img_width]),
                transforms.ToTensor()
        ]))

        self.train_ds, self.val_ds, self.test_ds = random_split(dataset, self.train_val_test_split)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        

