import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Load the stop sign images into an ImageFolder dataset
data_dir = '/path/to/stop_signs'
dataset = ImageFolder(data_dir)

# Split the dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Load the datasets into DataLoader instances for easier access
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
