import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from glob import glob
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize, ColorJitter,RandomRotation,RandomHorizontalFlip,RandomCrop, Normalize
from torchvision.models.quantization import mobilenet_v3_large

import numpy
from tqdm import tqdm
from PIL import Image
import argparse
from azureml.core import Run

# add run to ml
run = Run.get_context()
# parser incoming parameters 
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, dest="data_folder" , default = "")
parser.add_argument("--num-epochs", type=int, dest="epochs", default = 100)
parser.add_argument("--batch-size", type=int, dest="batch_size", default=64)
parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=1e-3)

args = parser.parse_args()
anchor_img_paths =  glob(os.path.join(args.data_folder, "anchor", "*.jpg") )
positive_img_paths =glob(os.path.join(args.data_folder, "positive", "*.jpg") )
negative_img_paths = glob(os.path.join(args.data_folder, "negative", "*.jpg") )
epochs = args.epochs
batch_size = args.batch_size

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
    
class EmbeddingNetworkModel(nn.Module):
    """
    A neural network model that generates embeddings for images.
    """
    def __init__(self):
        super(EmbeddingNetworkModel, self).__init__()
        self.feature_extractor = self._build_feature_extractor()

    def _build_feature_extractor(self):
        """
        Builds the feature extraction backbone of the network, which in this case is an EfficientNet-V2-L model.
        """
        backbone =model_fe = mobilenet_v3_large(pretrained=True, progress=True, quantize=False)
        output_layer = self._build_feature_projection_layer()
        final_model = nn.Sequential(
            backbone.quant,
            backbone.features,
            backbone.avgpool,
            backbone.dequant,
            nn.Flatten(1),
            output_layer
        )
        return final_model

    def _build_feature_projection_layer(self):
        """
        Builds the feature projection layer, which maps the feature vector to a higher-dimensional representation.
        """
        return nn.Sequential(
            nn.Linear(in_features=960, out_features=1024,bias=True),
            nn.Hardswish(inplace=True)
        )

    def forward_single_image(self, image):
        """
        Generates an embedding for a single input image.
        """
        embedding = self.feature_extractor(image)
        return embedding.float() 

    def forward(self, anchor_images, positive_images, negative_images):
        """
        Generates embeddings for a batch of anchor, positive, and negative images.
        """
        anchor_embeddings = self.forward_single_image(anchor_images)
        positive_embeddings = self.forward_single_image(positive_images)
        negative_embeddings = self.forward_single_image(negative_images)
        return anchor_embeddings, positive_embeddings, negative_embeddings
    
class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Parameters:
    patience (int): How long to wait after last time validation loss improved.
    verbose (bool): If True, prints a message for each validation loss improvement.
    save_model_path (str): Directory path to save the best-performing model checkpoint.
    """
    def __init__(self, patience: int = 10, verbose: bool = False, save_model_path: str = "SaveModels"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_model_path = save_model_path
        self.setup_folder()

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Call the object to monitor training and save the model if validation loss improves.
        
        Parameters:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): Model to be saved.
        """
        score = -val_loss  # Higher scores are better
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def setup_folder(self):
        """
        Create the directory to save the model checkpoint if it doesn't exist.
        """
        os.makedirs(self.save_model_path, exist_ok=True)
        
    def save_latest_model(self, model: torch.nn.Module):
        """
        Save the latest trained model.
        
        Parameters:
        model (torch.nn.Module): Model to be saved.
        """
        torch.save(model.state_dict(), os.path.join(self.save_model_path, 'latest_model.pt'))

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """
        Save the model checkpoint.
        
        Parameters:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): Model to be saved.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join("user_logs", 'best_model.pt'))
        
def train_valid_test_split(anchor_img_paths :list,
                           positive_img_paths :list,
                           negative_img_paths :list,
                           batch_size : int,
                           input_size : tuple,
                           train_rate :float,
                           valid_rate :float) -> tuple :
    # Calculate the thresshold values
    trainThressHold = int(len(positive_img_paths) * train_rate)
    validThressHold = int(trainThressHold + len(positive_img_paths) * valid_rate)

    # Set up transform
    train_transform = Compose([
        ToTensor(),
        Resize(input_size),
        RandomCrop(200, padding=4),
        RandomHorizontalFlip(0.5),
        RandomRotation(20),
        ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.1),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])
    test_valid_transform = Compose([
        ToTensor(),
        Resize(input_size),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = LoadDataset(anchor_img_paths[:trainThressHold],
                                positive_img_paths[:trainThressHold],
                                negative_img_paths[:trainThressHold],
                                train_transform)
    valid_dataset = LoadDataset(anchor_img_paths[trainThressHold:validThressHold],
                                positive_img_paths[trainThressHold:validThressHold],
                                negative_img_paths[trainThressHold:validThressHold],
                                test_valid_transform)
    test_dataset = LoadDataset(anchor_img_paths[validThressHold:],
                               positive_img_paths[validThressHold:],
                               negative_img_paths[validThressHold:],
                               test_valid_transform)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader

def getTotalTruePredict(anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative__embedding: torch.Tensor,
                epsilon: float = 0) -> list:
    positive_distance = torch.linalg.norm(anchor_embeddings - positive_embeddings, dim=1)
    negative_distance = torch.linalg.norm(anchor_embeddings - negative__embedding, dim=1)
    accuracies = (positive_distance + epsilon < negative_distance ).int()
    return torch.sum(accuracies)

def evaluateValidData(model, criterion, valid_dataloader) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loss = 0.0
    val_accuracy = 0
    sample_num = 0
    for i, (anchor_imgs, positive_imgs, negative_imgs)in enumerate(valid_dataloader):
        anchor_imgs = anchor_imgs.to(device)
        positive_imgs = positive_imgs.to(device)
        negative_imgs = negative_imgs.to(device)
        with torch.no_grad():
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            val_loss += loss.item()
            val_accuracy += getTotalTruePredict(anchor_embeddings, positive_embeddings, negative_embeddings)
            sample_num += len(anchor_imgs)
    val_average_loss = val_loss / (i+1)
    val_average_accuracy = val_accuracy / sample_num
    
    return val_average_loss, val_average_accuracy

def train(train_dataloader, valid_dataloader, test_dataloader, epochs, learning_rate) :
    print("------------------------------------------------START TRAINING------------------------------------------------")
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(os.getcwd())
    
    model = EmbeddingNetworkModel().to(device )
    early_stopping = EarlyStopping(patience=100, verbose=True)
    criterion = nn.TripletMarginLoss(margin=1,p=2,swap=True)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        train_progress = tqdm(train_dataloader,colour="yellow")
        total_loss = 0.0
        total_true_predict = 0
        sample_num = 0
        for i , (anchor_imgs, positive_imgs, negative_imgs) in enumerate(train_progress):
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)

            # Forward
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate loss
            cur_loss = loss.item()
            total_loss += cur_loss
                            
            # Calculate accuracy
            cur_true_predict = getTotalTruePredict(anchor_embeddings, positive_embeddings, negative_embeddings)
            sample_num += len(anchor_imgs)
            cur_accuracy = cur_true_predict / len(anchor_imgs)
            total_true_predict += cur_true_predict
            train_progress.set_description("Epoch {}/{} Train Loss: {:0.4f}, Train Accuracy: {:0.2f}".format(epoch+1, epochs, cur_loss, cur_accuracy ))
            
        model.eval()
        val_average_loss, val_average_accuracy = evaluateValidData(model, criterion, valid_dataloader)
        train_average_loss = total_loss / (i+1)
        train_average_accuracy = total_true_predict / sample_num
        print("Average Train Loss: {:0.4f}, Average Train Accuracy: {:0.2f}%".format(train_average_loss, train_average_accuracy*100),end=" ")
        print("Average Val Loss: {:0.4f}, Average Val Accuracy: {:0.2f}%".format(val_average_loss, val_average_accuracy*100))
        
        # early stopp 
        early_stopping(val_average_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

# add parser


# data load
train_dataloader, valid_dataloader, test_dataloader = train_valid_test_split(
    anchor_img_paths=anchor_img_paths,
    positive_img_paths=positive_img_paths,
    negative_img_paths=negative_img_paths,
    batch_size=batch_size,
    input_size=224,
    train_rate=0.8,
    valid_rate=0.1
)

train(train_dataloader, valid_dataloader, test_dataloader, epochs=args.epochs, learning_rate=args.learning_rate)