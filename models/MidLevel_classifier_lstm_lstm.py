import torch
from torch import nn
from utils.logger import logger


class FullyConnectedFusion(nn.Module):
    def __init__(self, num_classes=20, dropout_prob=0.6,hidden_size=512):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2500, 512)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        # 512 for sums and maximum
        self.fc3 = nn.Linear(512, num_classes)
        #1024 for concatenations
        #self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):

        feature_rgb, feature_emg = x
        feature_emg = self.flatten(feature_emg)
        feature_emg = self.relu(self.fc1(feature_emg))
        combined_feat = torch.maximum(feature_emg,feature_rgb.squeeze(dim=1))
        #combined_feat = torch.cat((feature_emg, feature_rgb.squeeze(dim=1)), dim=1)
        #combined_feat = torch.cat((feature_emg*0.2, feature_rgb.squeeze(dim=1)*0.8), dim=1)
        #combined_feat=0.2*feature_emg+0.8*feature_rgb.squeeze(dim=1)
        #combined_feat=feature_emg+feature_rgb.squeeze(dim=1)
        #combined_feat = torch.maximum(feature_emg,feature_rgb.squeeze(dim=1))
        
        combined_feat = self.dropout(combined_feat)
        results = self.fc3(combined_feat)

        return results, {}
