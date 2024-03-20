from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class LeNet5(nn.Module):

    # network structure
    def __init__(self,num_classes=20):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(768, 64)  # Adjusted based on input size
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        print(x.shape)
        x=x.unsqueeze(dim=1)
        x=x.permute(0,1,3,2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))  # Adjusted based on input size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x,{}
    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size) 
