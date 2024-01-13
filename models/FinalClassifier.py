from torch import nn


class MLP_classifier(nn.Module):
    def __init__(self, num_classes=8, input_size=1024):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        print(f"input shape: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        print(f"final shape: {x.shape}")
        return x, {}

class MLP_aggregation_classifier(nn.Module):
    def __init__(self, num_classes=8, input_size=1024, sequence_length=5):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for averaging along the sequence dimension

    def forward(self, x):

        x = self.avg_pool(x.permute(0, 2, 1))  # Permute for AdaptiveAvgPool1d and then restore the original order
        x = x.permute(0, 2, 1)
        x = self.fc1(x.squeeze(dim=1))  # Squeeze the sequence dimension
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x, {}
