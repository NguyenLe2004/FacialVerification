import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter, RandomRotation, RandomHorizontalFlip
from PIL import Image

class LoadDataset(Dataset):
    def __init__(self, anchor_paths, positive_paths, negative_paths, transform=None):
        self.anchor_paths = anchor_paths
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths
        self.transform = transform

    def __getitem__(self, index):
        anchor_image = self.transform(Image.open(self.anchor_paths[index]))
        positive_image = self.transform(Image.open(self.positive_paths[index]))
        negative_image = self.transform(Image.open(self.negative_paths[index]))
        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.anchor_paths)