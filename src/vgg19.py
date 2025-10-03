import torch
import torch.nn as nn


class VGG19(nn.Module):
    """
    Implementation of the VGG19 architecture with feature loss calculation.

    :param in_channels: Number of input channels for the first convolutional layer.
    :type in_channels: int
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.n_convs = 16
        self.chann = 64

        self.conv1 = nn.Conv2d(in_channels, self.chann, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.chann, self.chann, 3, 1, 1)

        self.conv3 = nn.Conv2d(self.chann, 2 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2 * self.chann, 2 * self.chann, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(2 * self.chann, 4 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(4 * self.chann, 4 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(4 * self.chann, 4 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(4 * self.chann, 4 * self.chann, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(4 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(8 * self.chann, 8 * self.chann, kernel_size=3, stride=1, padding=1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def calculate_feature_loss(self, x):
        """
        Calculates the feature loss between the reconstruction and source.

        :param x: Input tensor concatenated along dimension 0 (reconstruction and source).
        :type x: torch.Tensor
        :return: Mean squared error loss between reconstruction and source.
        :rtype: torch.Tensor
        """
        reconstruction, source = x.chunk(2, dim=0)
        return torch.mean((reconstruction - source) ** 2)

    def forward(self, x):
        """
        Forward pass through the VGG19 model. Calculates the feature loss at each layer.

        :param x: Input tensor concatenated along dimension 0 (reconstruction and source).
        :type x: torch.Tensor
        :return: Average feature loss across all convolutional layers.
        :rtype: torch.Tensor
        """
        # x is concatenated at dim 0 reconstruction from resvqvae and source
        x = self.conv1(x)
        loss = self.calculate_feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.mp(self.relu(x))

        x = self.conv3(x)
        loss += self.calculate_feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.mp(self.relu(x))

        x = self.conv5(x)
        loss += self.calculate_feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.mp(self.relu(x))

        x = self.conv9(x)
        loss += self.calculate_feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.mp(self.relu(x))

        x = self.conv13(x)
        loss += self.calculate_feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.calculate_feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.calculate_feature_loss(x)

        return loss / self.n_convs
