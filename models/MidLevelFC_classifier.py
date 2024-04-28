import torch
from torch import nn
from utils.logger import logger


class FullyConnectedFusion(nn.Module):
    def __init__(self, num_classes=20, dropout_prob=0.5):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5000, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for averaging along the sequence dimension

    def forward(self, x):

        feature_rgb, feature_emg = x
        feature_emg = self.flatten(feature_emg)
        feature_emg = self.fc1(feature_emg)

        concatenated_feat = torch.cat((feature_emg, feature_rgb.squeeze(dim=1)), dim=1)
        logger.info(f"FEATURE: {concatenated_feat.shape}")
        concatenated_feat = self.dropout(concatenated_feat)
        concatenated_feat = self.fc2(concatenated_feat)
        results = self.fc3(concatenated_feat)

        return results, {}