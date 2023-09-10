from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from lightning.pytorch import LightningDataModule

class kickBallDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, train_val_test_split, img_height, img_width, data_mean, data_std):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.img_height = img_height
        self.img_width = img_width
        self.data_mean = data_mean
        self.data_std = data_std


    def setup(self, stage):

        train_transform = transforms.Compose([
                transforms.Normalize(self.data_mean, self.data_std),
                transforms.Resize([self.img_height, self.img_width]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
                transforms.Resize([self.img_height, self.img_width]),
                transforms.ToTensor()
        ])

        # Load dataset
        dataset = datasets.ImageFolder(self.data_dir)

        # Split dataset into training, validation and test
        self.train_ds, self.val_ds, self.test_ds = random_split(dataset, self.train_val_test_split)

        # Apply train transform with data augmentation.
        self.train_ds.dataset.transform = train_transform

        # Apply val and test transform with no data augmentation. 
        self.val_ds.dataset.transform = val_transform
        self.test_ds.dataset.transform = val_transform


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
        


