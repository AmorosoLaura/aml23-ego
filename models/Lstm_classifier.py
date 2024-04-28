from torch import nn
from utils.logger import logger



class Lstm_classifier(nn.Module):
    def __init__(self, num_classes=20, input_size=1024, hidden_size=512, dropout_prob=0.5):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        out, _ = self.lstm(x)
        mid_level_features = {}
        mid_level_features['features'] = out
        out = self.dropout(out)
        out = self.relu(out)

        out = self.fc(out[:, -1, :]) # extract the last output of the sequence (the one obtained after all the timesteps)

        return out, mid_level_features