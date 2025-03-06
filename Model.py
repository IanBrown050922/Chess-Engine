import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN used to learn/give position evaluations
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__() # nn.Module constructor
        # 12 channels for 6 white piece types and 6 black piece types
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128) # 64 channels, each 8x8
        self.dropout = nn.Dropout(0.2) # dropout to prevent overfitting
        self.fc2 = nn.Linear(in_features=128, out_features=1) # output is a single evaluation score


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten for fully-connected layers; size -1 is inferred from other dimensions
        # result is x.size(0) many batches of flat tensors whose dimension is the product of the remaining dimensions of x
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x))) # dropout between fc layers
        x = self.fc2(x)
        return torch.tanh(x) # output values from -1 to 1