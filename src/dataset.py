from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from lightning.pytorch import LightningDataModule

class kickBallDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, img_height, img_width, data_mean, data_std):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_height = img_height
        self.img_width = img_width
        self.data_mean = data_mean
        self.data_std = data_std    


    def setup(self, stage):

        train_transform = transforms.Compose([
                transforms.Resize([self.img_height, self.img_width]),
               # transforms.RandomHorizontalFlip(p=0.5),
               # transforms.RandomVerticalFlip(p=0.5),
               # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
               # transforms.RandomRotation(degrees=(30, 70)),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_std),
        ])

        val_transform = transforms.Compose([
                transforms.Resize([self.img_height, self.img_width]),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean, self.data_std),
        ])


        self.train_ds = datasets.ImageFolder(self.data_dir + 'train', transform=train_transform)
        self.val_ds = datasets.ImageFolder(self.data_dir + 'val', transform=val_transform)
        self.test_ds = datasets.ImageFolder(self.data_dir + 'test', transform=val_transform)


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
        


