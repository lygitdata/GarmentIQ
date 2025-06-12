import torch.nn as nn
import timm


class CNN3(nn.Module):
    """
    CNN3 is a convolutional neural network designed for image classification with moderate depth.

    It consists of three convolutional blocks followed by fully connected layers.

    Args:
        num_classes (int): Number of output classes for classification.

    Attributes:
        features (nn.Sequential): Convolutional feature extractor.
        classifier (nn.Sequential): Fully connected classifier.
    """

    def __init__(self, num_classes):
        """
        Initializes the CNN3 model architecture.

        Args:
            num_classes (int): Number of target classes.
        """
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        Defines the forward pass of the CNN3 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN4(nn.Module):
    """
    CNN4 is a deeper convolutional neural network with four convolutional blocks.

    This architecture is designed for improved feature extraction by adding an extra conv block
    and adaptive pooling before classification layers.

    Args:
        num_classes (int): Number of output classes for classification.

    Attributes:
        features (nn.Sequential): Convolutional feature extractor with four blocks.
        classifier (nn.Sequential): Fully connected classifier.
    """

    def __init__(self, num_classes):
        """
        Initializes the CNN4 model architecture.

        Args:
            num_classes (int): Number of target classes.
        """
        super(CNN4, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            # Block 4 (Additional block for more depth)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Replace MaxPool with adaptive pooling to fix feature map size
            nn.AdaptiveAvgPool2d((4, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Defines the forward pass of the CNN4 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class tinyViT(nn.Module):
    """
    tinyViT is a vision transformer model based on the DeiT tiny architecture, pretrained on ImageNet.

    It replaces the original classification head with a linear layer matching the number of classes.

    Args:
        num_classes (int): Number of output classes.
        img_size (int): Input image size (assumed square).
        patch_size (int): Size of the image patches used by the transformer.

    Attributes:
        backbone (nn.Module): The underlying vision transformer model with modified head.
    """

    def __init__(self, num_classes, img_size, patch_size):
        """
        Initializes the tinyViT model with a pretrained DeiT backbone.

        Args:
            num_classes (int): Number of target classes.
            img_size (int): Input image size (height and width).
            patch_size (int): Patch size for the transformer.
        """
        super(tinyViT, self).__init__()
        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
        )
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the tinyViT model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, img_size, img_size).

        Returns:
            torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)
