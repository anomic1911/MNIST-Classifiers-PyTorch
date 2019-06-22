import torch
import torch.nn as nn
import numpy as np 
import torch.functional as F

# Modified LeNet
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1 ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fclayer = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
        ) 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x=x.view(-1, 16*5*5)
        x=self.fclayer(x)
        return x
