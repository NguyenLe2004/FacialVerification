import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataloader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def build_lr_model(train_dataloader: Dataloader, 
                   test_dataloader: Dataloader, 
                   model_path: str, 
                   save_path: str) -> None:
    """
    Builds and trains a logistic regression model using the given training and testing dataloaders.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for the testing data.
        model_path (str): Path to the pre-trained model.
        save_path (str): Path to save the trained logistic regression model.

    Returns:
        None
    """
    model = load_model(model_path=model_path)
    train_distance_df = get_distance_df(model, train_dataloader)
    test_distance_df = get_distance_df(model, test_dataloader)
    X_train, X_test, y_train, y_test = preprocess(train_df=train_distance_df, test_df=test_distance_df)
    lr_model = train_lr_model(X_train, y_train)
    evaluate(lr_model, X_test, y_test)
    save_model(lr_model, save_path=save_path)

def load_model(model_path: str) -> nn.Module:
    """
    Loads a pre-trained model from the given path.

    Args:
        model_path (str): Path to the pre-trained model.

    Returns:
        torch.nn.Module: The loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device).eval()
    return model

def save_model(model: nn.Module, save_path: str) -> None:
    """
    Saves the given model to the specified path.

    Args:
        model (torch.nn.Module): The model to be saved.
        save_path (str): Path to save the model.

    Returns:
        None
    """
    pickle.dump(model, open(save_path, 'wb'))

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Preprocesses the training and testing data.

    Args:
        train_df (pandas.DataFrame): Dataframe containing the training data.
        test_df (pandas.DataFrame): Dataframe containing the testing data.

    Returns:
        tuple: A tuple containing the preprocessed X_train, X_test, y_train, and y_test.
    """
    X_train = train_df["distance"].values.reshape(-1, 1)
    X_test = test_df["distance"].values.reshape(-1, 1)
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    return X_train, X_test, y_train, y_test

def evaluate(model: nn.Module, X: np.ndarray, y : np.ndarray) -> None:
    """
    Evaluates the given model on the provided data.

    Args:
        model (sklearn.linear_model.LogisticRegression): The model to be evaluated.
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The ground truth labels.

    Returns:
        None
    """
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))

def train_lr_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Trains a logistic regression model on the given training data.

    Args:
        X_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training labels.

    Returns:
        sklearn.linear_model.LogisticRegression: The trained logistic regression model.
    """
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def get_distance(first_embedding: torch.Tensor, second_embedding: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Euclidean distance between two embeddings.

    Args:
        first_embedding (torch.Tensor): The first embedding.
        second_embedding (torch.Tensor): The second embedding.

    Returns:
        torch.Tensor: The Euclidean distance between the two embeddings.
    """
    return torch.linalg.norm(first_embedding - second_embedding, dim=1)

def get_distance_df(model: nn.Module, dataloader: Dataloader) -> pd.DataFrame:
    """
    Computes the distance between anchor, positive, and negative embeddings for a given dataloader.

    Args:
        model (torch.nn.Module): The pre-trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the data.

    Returns:
        pandas.DataFrame: A dataframe containing the distance and label for each sample.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_df = pd.DataFrame({
        "distance": [],
        "label": []
    })
    with torch.no_grad():
        progress_bar = tqdm(dataloader)
        for anchor_imgs, positive_imgs, negative_imgs in progress_bar:
            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)

            # Forward
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
            positive_distance = get_distance(anchor_embeddings, positive_embeddings)
            negative_distance = get_distance(anchor_embeddings, negative_embeddings)

            distance_df = pd.concat([distance_df, pd.DataFrame({
                "distance": [d.item() for d in positive_distance] + [d.item() for d in negative_distance],
                "label": [1] * len(positive_distance) + [0] * len(negative_distance)
            })], ignore_index=True)
            progress_bar.set_description("running.....")
    return distance_df