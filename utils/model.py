import pickle
import os
import torch 
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v3_large, MobileNet_V3_Large_QuantizedWeights

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
        backbone = mobilenet_v3_large(MobileNet_V3_Large_QuantizedWeights,pretrained=True, progress=True, quantize=False)
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
            nn.Hardswish(inplace=True),
        )

    def forward_single_image(self, image : torch.Tensor):
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

class FacialVerification(nn.Module):
    """
    A PyTorch module for facial verification.
    
    Attributes:
        feature_extract (EmbeddingNetworkModel): A pre-trained feature extraction model.
        coef (float): Coefficient for the logistic regression model.
        intercept (float): Intercept for the logistic regression model.
    """
    def __init__(self):
        super().__init__()
        self.feature_extract_model = torch.jit.load("SaveModels/feature_extract_best_model.pth").eval()
        for param in self.feature_extract_model.parameters():
            param.requires_grad = False
        lr_model = pickle.load(open("SaveModels/ClassifyModel.pkl","rb"))
        self.coef = lr_model.coef_[0][0]
        self.intercept = lr_model.intercept_[0]

    def get_distance(self, verify_encode, base_encode):
        """
        Calculates the Euclidean distance between the input encodings.
        
        Args:
            verify_encode (torch.Tensor): Encoding of the verification image.
            base_encode (torch.Tensor): Encoding of the base image.
        
        Returns:
            torch.Tensor: Euclidean distance between the input encodings.
        """
        return torch.linalg.norm(base_encode - verify_encode, dim=1)
    
    def get_predict(self, distance):
        """
        Calculates the prediction probability using a logistic regression model.
        
        Args:
            distance (torch.Tensor): Euclidean distance between the input encodings.
        
        Returns:
            torch.Tensor: Prediction probability.
        """
        return 1 / (1 + torch.exp(- self.intercept - self.coef * distance))
    
    @torch.jit.export
    def extract_feature(self, image):
        """
        Extracts the feature representation from the given image using the feature extraction model.

        Args:
            image (torch.Tensor): The input image tensor. The tensor should have shape (C, H, W), where C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The extracted feature representation.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        feature_extract = self.feature_extract_model.forward_single_image(image)
        return feature_extract
    
    @torch.jit.export
    def verify(self, verify_image, base_encode):
        """
        Performs facial verification by comparing the input images.
        
        Args:
            verify_images (torch.Tensor): Tensor of verification images.
            base_image (torch.Tensor): Tensor of the base image.
            
        Returns:
            torch.Tensor: Prediction probabilities for the verification images.
        """
        if verify_image.ndim == 3:
            verify_image = verify_image.unsqueeze(0)
        verify_encode = self.feature_extract_model.forward_single_image(verify_image)
        distance = self.get_distance(verify_encode, base_encode)
        output = self.get_predict(distance)
        return output
    
    def forward(self, verify_image, base_image):
        """
        Performs facial verification by comparing the input images.
        
        Args:
            verify_images (torch.Tensor): Tensor of verification images.
            base_image (torch.Tensor): Tensor of the base image.
            
        Returns:
            torch.Tensor: Prediction probabilities for the verification images.
        """
        if verify_image.ndim == 3:
            verify_image = verify_image.unsqueeze(0)
        if base_image.ndim == 3:
            base_image = base_image.unsqueeze(0)
        base_encode = self.extract_feature(base_image)
        verify_encode = self.extract_feature(verify_image)
        distance = self.get_distance(verify_encode, base_encode)
        output = self.get_predict(distance)
        return output