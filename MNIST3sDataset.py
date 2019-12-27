import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from PIL import Image

class MNIST3sDataset(data.Dataset):
  def __init__(self, directory, transforms):
    self.directory = directory
    self.filenames = os.listdir(directory)
    self.transforms = transforms

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    img = Image.open(os.path.join(self.directory, self.filenames[idx]))
    img = self.transforms(img)
    return img
