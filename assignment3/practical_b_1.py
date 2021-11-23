import torch
from torch.utils.data import Dataset

from load_mnist import load_mnist
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, type, label):
        self.images, self.labels = load_mnist(dataset=type,
                                              path="data/FashionMNIST/raw")
        mask = self.labels.eq(label)
        self.labels = self.labels[mask]
        self.images = self.images[mask]
        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class SortedImageDataset(Dataset):
    def __init__(self, type):

        self.images, self.labels = load_mnist(dataset=type,
                                              path="data/FashionMNIST/raw")
        zipped = zip(self.images, self.labels)
        sorted_ = sorted(zipped,key=lambda x: x[1])
        self.images, self.labels = zip(*sorted_)
        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class NormalDataset(Dataset):
    def __init__(self, type):
        self.images, self.labels = load_mnist(dataset=type,
                                              path="data/FashionMNIST/raw")
      

        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.images[index], self.labels[index]