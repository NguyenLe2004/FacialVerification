import torch
import torch.nn as nn
class NPLBLoss(nn.Module):
    def __init__(self, margin=1.0,beta=1.0):
        super(NPLBLoss, self).__init__()
        self.margin = margin
        self.beta = beta
    def forward(self, anchor_embeddings: torch.Tensor, positive_embeddings: torch.Tensor, negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the No Pairs Left Behind loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embedding_dim).
            positive (torch.Tensor): Positive embeddings, shape (batch_size, embedding_dim).
            negative (torch.Tensor): Negative embeddings, shape (batch_size, embedding_dim).
            
        Returns:
            torch.Tensor: Mean loss.
        """
        # Compute distances
        d_anc_pos = torch.linalg.norm(anchor_embeddings - positive_embeddings, dim=1)
        d_anc_neg = torch.linalg.norm(anchor_embeddings - negative_embeddings, dim=1)
        d_neg_pos = torch.linalg.norm(anchor_embeddings - negative_embeddings, dim=1)

        # Compute the loss
        loss = torch.max(torch.tensor(0.0), d_anc_pos - d_anc_neg + self.margin) + self.beta * torch.square(d_neg_pos - d_anc_neg)
        
        return loss.mean()