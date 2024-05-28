import torch.nn as nn
import torch


class BaselineModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # assuming image size is 320x320
        # predict probs for each pixel
        self.m = nn.Sequential(
            nn.Conv2d(3, 128, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 512, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 320 * 320 * 2),
            nn.Unflatten(1, (2, 320, 320)),
            nn.Softmax(dim=1),
        )

        torch.nn.init.xavier_uniform_(self.m[0].weight)

    def forward(self, x):
        return self.m(x)


class FullConvolutionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        # assuming image size is 320x320
        # predict probs for each pixel
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(256, 512, 2, 1),
            nn.Flatten(),
            nn.Linear(512, 320 * 320 * 2),
            nn.Unflatten(1, (2, 320, 320)),
            nn.Softmax(dim=1),
        )

        torch.nn.init.xavier_uniform_(self.m[0].weight)

    def forward(self, x):
        return self.m(x)
