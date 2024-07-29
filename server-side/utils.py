from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

def preprocess_image(image: Image, target_size: int = 224) -> torch.Tensor:
    """
    Preprocesses the input image by applying various transformations.

    Args:
        image (PIL.Image): The input image.
        target_size (int): The desired size of the output image. Default is 224.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = Compose([
        ToTensor(),
        Resize((target_size, target_size)),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    image_rgb = image.convert('RGB')
    return transform(image_rgb)

def load_pretrained_model(checkpoint_path: str) -> torch.nn.Module:
    """
    Loads a pre-trained PyTorch model from the specified checkpoint.

    Args:
        checkpoint_path (str): The path to the model checkpoint.

    Returns:
        torch.nn.Module: The loaded pre-trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint_path, map_location=device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def crop_image_with_rates(image: Image, height_rate: float, width_rate: float) -> Image:
    """
    Crops the input image based on the specified height and width rates.

    Args:
        image (PIL.Image): The input image.
        height_rate (float): The rate of the height to be cropped.
        width_rate (float): The rate of the width to be cropped.

    Returns:
        PIL.Image: The cropped image.
    """
    width, height = image.size
    left = int(width * (1 - width_rate) / 2)
    top = int(height * (1 - height_rate) / 2)
    right = int(width * (1 + width_rate) / 2)
    bottom = int(height * (1 + height_rate) / 2)
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def preprocess_input_image(image: Image) -> torch.Tensor:
    """
    Preprocesses the input image by cropping and transforming it.

    Args:
        image (PIL.Image): The input image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    cropped_image = crop_image_with_rates(image=image, height_rate=0.9, width_rate=0.6)
    preprocessed_image = preprocess_image(image=cropped_image)
    return preprocessed_image

def calculate_distance(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> float:
    """
    Calculates the L2 distance between two PyTorch tensors.

    Args:
        tensor_1 (torch.Tensor): The first tensor.
        tensor_2 (torch.Tensor): The second tensor.

    Returns:
        float: The L2 distance between the two tensors.
    """
    distance = torch.linalg.norm(tensor_1 - tensor_2, dim=1)
    return distance.item()