"""
File: train.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import wandb

wandb.init(project='cs231n-final-project')

class BaselineModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BaselineModel, self).__init__()
        self.num_classes = num_classes


        # assuming image size is 640x640
        # predict probs for each pixel
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 20 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 640 * 640)
        )

    def forward(self, x):
        return F.sigmoid(self.m(x))


if __name__ == '__main__':
