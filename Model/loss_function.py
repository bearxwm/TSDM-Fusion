import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models import VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class GradientConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GradientConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        kernel = [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]

        kernel = torch.tensor(kernel).float().expand([out_dim, in_dim, 3, 3])

        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
        x1 = pad(x)
        x1 = F.conv2d(x1, self.weight)

        return x1


class Gradloss(nn.Module):

    def __init__(self, in_dim=3):
        super(Gradloss, self).__init__()

        self.source_conv = GradientConv(in_dim=in_dim, out_dim=in_dim)
        self.fuse_conv = GradientConv(in_dim=in_dim, out_dim=in_dim)

    def forward(self, input_image, output_image):
        x1 = self.source_conv(input_image)
        x2 = self.fuse_conv(output_image)

        grad_loss = torch.mean(torch.pow((x1 - x2), 2))

        return grad_loss


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
         <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
        <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block
        <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        content_loss = F.l1_loss(sr_feature, hr_feature)

        return content_loss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss