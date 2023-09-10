import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(csv_file):

    df = pd.read_csv(csv_file)

    fig, ax1 = plt.subplots()

    sns.lineplot(data=df, x='step', y='val_loss', ax=ax1, label='Validation Loss', color='b', linewidth=2)
    sns.lineplot(data=df, x='step', y='train_loss', ax=ax1, label='Training Loss', color='b', linewidth=2, linestyle='dashed')

    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='step', y='val_accuracy', ax=ax2, label='Validation Accuracy', color='g', linewidth=2)
    sns.lineplot(data=df, x='step', y='train_accuracy', ax=ax2, label='Training Accuracy', color='g', linewidth=2, linestyle='dashed')

    ax1.set_ylabel('Loss', color='b')
    ax2.set_ylabel('Accuracy', color='g')

    ax1.set_ylim(0, df['train_loss'].max()*1.1)
    ax2.set_ylim(0, 1)

    ax1.legend(loc='upper right')
    ax2.legend(loc='lower right') 

    plt.title("Losses and Accuracy")

    plt.show()

from torchvision import datasets, transforms
from torch import zeros
from torch.utils.data import DataLoader
from tqdm import tqdm

def calc_mean_std(data_dir, batch_size, num_workers, num_channels):

    dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
    # Step 3: Create a DataLoader with pin_memory for data loading optimization
    full_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    # Step 4: Initialize variables for mean and std
    mean = zeros(num_channels)
    std = zeros(num_channels)

    # Step 5: Calculate the mean and std for the dataset in parallel
    for inputs, _ in tqdm(full_loader, desc="==> Computing mean and std"):
        # Step 6: Compute mean and std for each channel using PyTorch operations
        mean += inputs.mean(dim=(0, 2, 3))
        std += inputs.std(dim=(0, 2, 3))

    # Step 7: Normalize the mean and std by dividing by the dataset size
    mean /= len(dataset)
    std /= len(dataset)

    # Step 8: Print the results
    print('MEAN', mean)
    print('STD', std)

#  utilities.calc_mean_std(
#      data_dir=config.DATA_DIR,
#      batch_size=config.BATCH_SIZE,
#      num_workers=config.NUM_WORKERS,
#      n_channels=config.NUM_CHANNELS   
#  )