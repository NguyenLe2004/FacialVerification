import os
import torch
import torch.nn as nn

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Parameters:
    patience (int): How long to wait after last time validation loss improved.
    verbose (bool): If True, prints a message for each validation loss improvement.
    save_model_path (str): Directory path to save the best-performing model checkpoint.
    """
    def __init__(self, patience: int = 10, verbose: bool = False,  delta : float =0, save_model_path: str = "SaveModels"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_model_path = save_model_path
        self.setup_folder()

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Call the object to monitor training and save the model if validation loss improves.
        
        Parameters:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): Model to be saved.
        """
        score = val_loss  # Higher scores are better
        
        if self.best_score is None:
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
        self.save_latest_model(model)
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
        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(self.save_model_path, 'feature_extract_last_model.pth'))

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """
        Save the model checkpoint.
        
        Parameters:
        val_loss (float): Current validation loss.
        model (torch.nn.Module): Model to be saved.
        """
        if self.verbose: 
            print(f'Validation loss decreased ({self.best_score} --> {val_loss}). Saving model...')
        self.best_score = val_loss
        model_scripted = torch.jit.script(model)
        model_scripted.save(os.path.join(self.save_model_path, 'feature_extract_best_model.pth'))