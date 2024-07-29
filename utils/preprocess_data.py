import os
import shutil
from glob import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter,RandomRotation,RandomHorizontalFlip
from utils.load_dataset import LoadDataset

def AnchorPositiveNegativeSplit(src_data_path : str, anchor_path :str, positive_path :str, negative_path :str) -> None :
    """
    Split the source data to Anchor, Positive and Negative groups then add to data folder.

    Args:
        src_data_path (str): The path to the directory containing the original image data.
        anchor_path (str): The path to the anchor directory.
        positive_path (str): The path to the positive directory.
        negative_path (str): The path to the negative directory.
    """
    negative_count = 1
    positive_anchor_count = 1
    for dir in os.listdir(src_data_path):
        dir_path = os.path.join(src_data_path, dir)
        img_paths = os.listdir(dir_path)
        img_num = len(img_paths)
        if img_num < 2:  # if number of person image is one then use it for negative
            old_path = os.path.join(dir_path, img_paths[0])
            new_path = os.path.join(negative_path, "negative_{}.jpg".format(negative_count))
            shutil.copy(old_path, new_path)
            negative_count += 1
        else: # if number of person image greater than one then use it for anchor and positive
            img_num = min(15, img_num)  # limit each person total images less than 15 img
            for i in range(img_num - 1):
                old_anchor_path = os.path.join(dir_path, img_paths[i])
                old_positive_path = os.path.join(dir_path, img_paths[i + 1])
                new_anchor_path = os.path.join(anchor_path, "anchor_{}.jpg".format(positive_anchor_count))
                new_positive_path = os.path.join(positive_path, "positive_{}.jpg".format(positive_anchor_count))
                shutil.copy(old_anchor_path, new_anchor_path)
                shutil.copy(old_positive_path, new_positive_path)
                positive_anchor_count += 1

def UpsampleNegative(negative_path : str,positive_or_anchor_path : str, ) -> None:
    """
    Randomly choose existing images until reaching the same number of images as the positive or anchor images.

    Args:
        negative_path (str): Path to the negative folder.
        positive_or_anchor_path (str): Path to the positive or anchor folder.
    """
    negative_paths = os.listdir(negative_path)
    positive_anchor_path = os.listdir(positive_or_anchor_path)
    negative_path_num = len(negative_paths)
    positive_anchor_path_num = len(positive_anchor_path)
    for i in range(positive_anchor_path_num - negative_path_num):
        random_index = random.randint(0, len(negative_paths) - 1)
        src_path = os.path.join(negative_path, negative_paths[random_index])
        cp_path = os.path.join(negative_path, "negative_{}.jpg".format(negative_path_num+1))
        shutil.copy(src_path, cp_path)
        negative_path_num += 1
    print("Number of negative images after upsample: {}".format(len(os.listdir(negative_path))))

def train_valid_test_split(anchor_img_paths :list,
                           positive_img_paths :list,
                           negative_img_paths :list,
                           batch_size : int,
                           input_size : tuple,
                           train_rate :float,
                           valid_rate :float) -> tuple[LoadDataset, LoadDataset, LoadDataset] :
    """
    Split the data into training, validation, and test sets.

    Parameters:
        anchor_img_paths (list): A list of paths to the anchor images.
        positive_img_paths (list): A list of paths to the positive images.
        negative_img_paths (list): A list of paths to the negative images.
        batch_size (int): The batch size for each data batch.
        input_size (tuple): Input size (width,height) of each image.
        train_rate (float): The proportion of data used for training.
        valid_rate (float): The proportion of data used for validation.

    Returns:
        tuple: A tuple containing the DataLoader objects for the training, validation, and test sets.
    """
    # Calculate the thresshold values
    trainThressHold = int(len(positive_img_paths) * train_rate)
    validThressHold = int(trainThressHold + len(positive_img_paths) * valid_rate)

    # Set up transform
    transform = {
        "train": Compose([
            ToTensor(),
            Resize(input_size),
            RandomHorizontalFlip(0.5),
            RandomRotation(30),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ]) ,
        "test": Compose([
            ToTensor(),
            Resize(input_size)
        ]),
        "valid" : Compose([
            ToTensor(),
            Resize(input_size)
        ]),
    }

    # Create datasets
    train_dataset = LoadDataset(anchor_paths = anchor_img_paths[:trainThressHold],
                                positive_paths = positive_img_paths[:trainThressHold],
                                negative_paths = negative_img_paths[:trainThressHold],
                                transform= transform["train"])
    valid_dataset = LoadDataset(anchor_paths = anchor_img_paths[trainThressHold:validThressHold],
                                positive_paths = positive_img_paths[trainThressHold:validThressHold],
                                negative_paths = negative_img_paths[trainThressHold:validThressHold],
                                transform = transform["valid"])
    test_dataset = LoadDataset(anchor_paths = anchor_img_paths[validThressHold:],
                               positive_paths = positive_img_paths[validThressHold:],
                               negative_paths = negative_img_paths[validThressHold:],
                               transform= transform["test"])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader

def DisplayExampleEachDataloader(dataloders : tuple[DataLoader, DataLoader, DataLoader]) -> None:
    """
    Displays an example from each DataLoader.

    Args:
        dataloaders (list): List of DataLoader instances.
    """
    fig, ax = plt.subplots(3, 3, figsize=(7, 10))
    for i, dataloader in enumerate(dataloders):
        batch = next(iter(dataloader))

        anchors, positives, negatives = batch
        anchor_image = anchors[0]
        positives_image = positives[0]
        negatives_image = negatives[0]

        anchor_image = anchor_image.numpy()
        positives_image = positives_image.numpy()
        negatives_image = negatives_image.numpy()

        anchor_image = np.transpose(anchor_image, (1, 2, 0))
        positives_image = np.transpose(positives_image, (1, 2, 0))
        negatives_image = np.transpose(negatives_image, (1, 2, 0))

        ax[i, 0].imshow(anchor_image)
        ax[i, 0].set_title("Anchor")
        ax[i, 1].imshow(positives_image)
        ax[i, 1].set_title("Positive")
        ax[i, 2].imshow(negatives_image)
        ax[i, 2].set_title("Negative")
    plt.suptitle("Example Images from Each DataLoader")
    plt.tight_layout()
    plt.show()

def PreprocessDataset(src_data_path: str, input_size: tuple, batch_size: int, isLoad: bool) -> tuple:
    """
    The function preprocesses a dataset for training.
    It organizes images into anchor, positive, and negative folders, upsamples the negative images,
    splits the data into train, validation, and test sets, and returns dataloaders for each set.

    Arguments:
        src_data_path (str): The path to the source data.
        input_size (Tuple[int, int]): The desired input size for the images.
        batch_size (int): The batch size for the dataloaders.
        isLoad (bool): Indicates whether to load data from the source directory.

    Returns:
        Tuple: A tuple containing the train dataloader, validation dataloader, and test dataloader.
    """
    # Set up paths and folders
    anchor_path = os.path.join("data", "anchor")
    positive_path = os.path.join("data", "positive")
    negative_path = os.path.join("data", "negative")

    os.makedirs(positive_path, exist_ok=True)
    os.makedirs(anchor_path, exist_ok=True)
    os.makedirs(negative_path, exist_ok=True)

    # Add data from source to created folders
    if isLoad:
        print("_______________________________________________________WAITING FOR LOADING_______________________________________________________")
        print("If you have already loaded data to data folder, use option `-nl` or set `isLoad = false` in the `PreprocessDataset` function.")
        AnchorPositiveNegativeSplit(src_data_path=src_data_path, anchor_path=anchor_path, positive_path=positive_path, negative_path=negative_path)

    # Upsample negative because number of negative images less than positive and anchor images
    UpsampleNegative(negative_path=negative_path, positive_or_anchor_path=positive_path)

    # Check if anchor, positive and negative folders created successfully
    anchor_img_paths = sorted(glob(os.path.join(anchor_path, "*.jpg")))
    positive_img_paths = sorted(glob(os.path.join(positive_path, "*.jpg")))
    negative_img_paths = sorted(glob(os.path.join(negative_path, "*.jpg")))

    print("Number of anchor image: {}".format(len(anchor_img_paths)))
    print("Number of positive image: {}".format(len(positive_img_paths)))
    print("Number of negative image: {}".format(len(negative_img_paths)))

    # Split data into train set, valid set and test set
    train_valid_test_dataloader = train_valid_test_split(
        anchor_img_paths=anchor_img_paths,
        positive_img_paths=positive_img_paths,
        negative_img_paths=negative_img_paths,
        batch_size=batch_size,
        input_size=input_size,
        train_rate=0.8,
        valid_rate=0.1
    )

    return train_valid_test_dataloader