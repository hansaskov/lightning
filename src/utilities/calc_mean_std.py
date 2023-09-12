from torchvision import datasets, transforms
from torch import zeros
from torch.utils.data import DataLoader
from tqdm import tqdm



def calc_mean_std(data_dir, batch_size, num_workers, num_channels):

    # Step 1: Define which dataset to calculate on
    dataset = datasets.ImageFolder(data_dir + "train", transform=transforms.ToTensor(),)
    # Step 2: Create a DataLoader with pin_memory for data loading optimization
    full_loader = DataLoader(dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    # Step 3: Initialize variables for mean and std
    mean = zeros(num_channels)
    std = zeros(num_channels)

    # Step 4: Calculate the mean and std for the dataset in parallel
    for inputs, _ in tqdm(full_loader, desc="==> Computing mean and std"):
        # Step 5: Compute mean and std for each channel using PyTorch operations
        mean += inputs.mean(dim=(0, 2, 3))
        std += inputs.std(dim=(0, 2, 3))

    # Step 6: Normalize the mean and std by dividing by the dataset size
    mean /= len(dataset)
    std /= len(dataset)

    # Step 7: Print the results
    print('MEAN', mean)
    print('STD', std)

    return mean, std

#  utilities.calc_mean_std(
#      data_dir=config.DATA_DIR,
#      batch_size=config.BATCH_SIZE,
#      num_workers=config.NUM_WORKERS,
#      num_channels=config.NUM_CHANNELS   
#  )