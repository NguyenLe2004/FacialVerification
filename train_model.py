import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.preprocess_data import PreprocessDataset
from utils.early_stop import EarlyStopping
from utils.model import EmbeddingNetworkModel, FacialVerification
from utils.loss import NPLBLoss
from utils.train_lr_model import build_lr_model
from warnings import filterwarnings

filterwarnings("ignore")

def main(args) :
    # Preprocess data
    train_dataloader, valid_dataloader, test_dataloader = PreprocessDataset(
        src_data_path = args.src_data_path,
        input_size = (args.input_size,args.input_size),
        batch_size = args.batch_size,
        isLoad = not args.nl,
    )
    # start train feature extract model
    train(train_dataloader, valid_dataloader, epochs=args.epochs, learning_rate=args.learning_rate)
    # evaluate on test data
    test(test_dataloader=test_dataloader)
    # build classify model use logistic regression
    build_lr_model(train_dataloader = train_dataloader,
                   test_dataloader = test_dataloader,
                   model_path = os.path.join("SaveModels", "feature_extract_best_model.pth"),
                   save_path = os.path.join("SaveModels", "ClassifyModel.pkl"))
    # build and save final model
    final_model = FacialVerification()
    save_model(model=final_model, save_path=os.path.join("SaveModels","FacialVerification.pth"))

def get_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--src_data_path","-sp", type=str, default="lfw", help="Path to source data")
    parser.add_argument("-nl",action="store_true", help ="Indicates whether to load data from the source directory")
    parser.add_argument("--input_size","-i", type=int, default=224)
    parser.add_argument("--batch_size","-b", type=int, default=64)
    parser.add_argument("--epochs","-e",type=int, default=100)
    parser.add_argument("--learning_rate","-lr",type=float, default=1e-3)
    args = parser.parse_args()
    return args

def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, learning_rate: float) -> None:
    """
    Train a model using the provided data loaders, number of epochs, and learning rate.

    Args:
        train_dataloader (DataLoader): The training data loader.
        valid_dataloader (DataLoader): The validation data loader.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingNetworkModel().to(device)
    early_stopping = EarlyStopping(patience=5, verbose=True, delta=1e-4)
    criterion = NPLBLoss(margin=1)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    print("------------------------------------------------START TRAINING------------------------------------------------")
    for epoch in range(epochs):
        ### TRAIN STEP
        model.train()
        optimizer.zero_grad()
        train_progress = tqdm(train_dataloader,colour="yellow")
        for anchor_imgs, positive_imgs, negative_imgs in train_progress:
            # push data to GPU
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

            # Calculate current accuracy
            cur_true_predict, cur_top_true_predict = calculate_total_true_predictions(anchor_embeddings, positive_embeddings, negative_embeddings)
            cur_accuracy = cur_true_predict / len(anchor_imgs)
            cur_top_accuracy = cur_top_true_predict / len(anchor_imgs)

            # set discription for progress bar
            train_progress.set_description("Epoch {}/{} Train Loss: {:0.4f}, Train Accuracy: {:0.2f}%, Train Top Accuracy: {:0.2f}%"
                                        .format(epoch+1, epochs, cur_loss, cur_accuracy*100, cur_top_accuracy*100 ))

        ### VALIDATION STEP
        model.eval()

        # Calculate validation loss and accuracy
        val_loss, val_accuracy, val_top_accuracy = evaluate_model(model, criterion, valid_dataloader)
        print("Val Loss: {:0.4f}, Val Accuracy: {:0.2f}%, Val Top Accuracy: {:0.2f}".format(val_loss, val_accuracy*100, val_top_accuracy*100))
        
        # Update the learning rate
        scheduler.step(val_loss)

        # Early stopp 
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def test(test_dataloader: DataLoader) -> tuple[float,float,float]:
    """
    Evaluate the model on the validation or test data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The validation or test data loader.
        device (torch.device): The device to use for evaluation.

    Returns:
        tuple[float,float,float] : the average loss, accuracy and top accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("SaveModels/feature_extract_best_model.pth",map_location=device).eval()
    criterion = NPLBLoss(margin=1)
    test_loss, test_accuracy, test_top_accuracy = evaluate_model(model, criterion,test_dataloader)
    print("Test Loss: {:0.4f}, Test Accuracy: {:0.2f}%, Test Top Accuracy: {:0.2f}".format(test_loss,test_accuracy*100, test_top_accuracy*100))
    
def save_model(model,save_path: str) -> None:
    """
    Save the model to the specified path.

    Args:
        model (torch.nn.Module): The model to be saved.
        save_path (str): The path to save the model.

    Returns:
        None
    """
    script = torch.jit.script(model)
    script.save(save_path)

def calculate_total_true_predictions(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embedding: torch.Tensor,
    epsilon: float = 0.0,
) -> tuple[int, int]:
    """
    Calculate the total number of correct and top predictions.
    
    Args:
        anchor_embeddings (torch.Tensor): Tensor of anchor embeddings.
        positive_embeddings (torch.Tensor): Tensor of positive embeddings.
        negative_embedding (torch.Tensor): Tensor of negative embedding.
        epsilon (float, optional): Epsilon value for top prediction. Defaults to 0.0.
    
    Returns:
        tuple[int, int]: Total number of correct predictions and total number of top predictions.
    """
    anchor_positive_distances = torch.linalg.norm(anchor_embeddings - positive_embeddings, dim=1)
    anchor_negative_distances = torch.linalg.norm(anchor_embeddings - negative_embedding, dim=1)
    positive_negative_distances = torch.linalg.norm(positive_embeddings - negative_embedding, dim=1)

    is_correct = (anchor_positive_distances < anchor_negative_distances ).int()
    is_top = ((anchor_positive_distances + epsilon) < anchor_negative_distances) & (anchor_positive_distances < positive_negative_distances).int()
    
    total_correct = torch.sum(is_correct)
    total_top = torch.sum(is_top)
    
    return total_correct, total_top

def evaluate_model(model: EmbeddingNetworkModel, criterion: nn.Module, dataloader: DataLoader) -> tuple[float, float, float]:
    """
    Evaluates the performance of the given model on the provided data loader.

    Args:
        model (EmbeddingNetworkModel): The model to be evaluated.
        criterion (nn.Module): The loss function used for evaluation.
        dataloader (DataLoader): The data loader containing the evaluation data.

    Returns:
        Tuple[float, float, float]: A tuple containing the average loss, average accuracy, and average top accuracy.
    """        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_loss = 0.0
    total_correct = 0
    total_top_correct = 0
    total_samples = 0
    progress_bar =tqdm(dataloader)
    for anchor_imgs, positive_imgs, negative_imgs in progress_bar:
        anchor_imgs = anchor_imgs.to(device)
        positive_imgs = positive_imgs.to(device)
        negative_imgs = negative_imgs.to(device)
        with torch.no_grad():
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchor_imgs, positive_imgs, negative_imgs)
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()

            num_correct, num_top_correct = calculate_total_true_predictions(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_correct += num_correct
            total_top_correct += num_top_correct
            total_samples += len(anchor_imgs)
        progress_bar.set_description("Running evaluate:...")

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_correct / total_samples
    average_top_accuracy = total_top_correct / total_samples

    return average_loss, average_accuracy, average_top_accuracy



if __name__ == "__main__" :
    args = get_args()
    main(args)