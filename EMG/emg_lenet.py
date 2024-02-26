from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
""" 
class LeNet5(nn.Module):
    def __init__(self, num_classes=20):  # Adjust output_size for your number of classes
        super().__init__()
        self.conv1 = nn.Conv2d(32, 6, kernel_size=(5, 5), padding=2)  # Input channels: 1 (grayscale)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6,16, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(16 * 1  * 12, 120)  # Adjust based on output shape of conv2
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x) # Flatten the output of conv2
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x """

#Defining the convolutional neural network
class LeNet5(nn.Module):

    # network structure
    def __init__(self,num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1   = nn.Linear(38336, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x=x.unsqueeze(dim=1)
        x=x.permute(0,1,3,2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
      
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)