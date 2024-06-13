import torch
from torch import nn
from utils.logger import logger


class FullyConnectedFusion(nn.Module):
    def __init__(self, num_classes=20, dropout_prob=0.5,hidden_size=512):
        super().__init__()

        self.flatten = nn.Flatten()
        #input dimesnion is 2500 in case of features 5 s    
        self.fc1 = nn.Linear(256, 50)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(100, 20)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for averaging along the sequence dimension

    def forward(self, x):

        feature_rgb, feature_emg = x
        feature_emg = self.flatten(feature_emg)
        feature_rgb = self.relu(self.fc1(feature_rgb))
        logger.info(f"feature rgb: {feature_rgb.shape}, feature emg: {feature_emg.shape}")
        combined_feat = torch.cat((feature_emg, feature_rgb.squeeze(dim=1)), dim=1)
        
        #combined_feat = torch.cat((feature_emg*0.2, feature_rgb.squeeze(dim=1)*0.8), dim=1)
        #combined_feat=0.2*feature_emg+0.8*feature_rgb.squeeze(dim=1)
        #combined_feat = torch.maximum(feature_emg,feature_rgb.squeeze(dim=1))
        
        combined_feat = self.dropout(combined_feat)
        #concatenated_feat = self.relu(self.fc2(concatenated_feat))
        results = self.fc3(combined_feat)

        return results, {}