from torch import nn


class Lstm_classifier(nn.Module):
    def __init__(self, num_classes=8, input_size=1024, batch_normalization=False):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size=512, num_layers=1, batch_first=True)
        self.use_batch = batch_normalization
        self.batch_norm = nn.BatchNorm1d(512)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        # Forward pass
        out, _ = self.lstm(x)
        if self.use_batch:
            out = out.permute(0, 2, 1)  # Permute to match the input shape of BatchNorm2d
            out = self.batch_norm(out)
            out = out.permute(0, 2, 1)  # Permute back to the original shape
        out = nn.ReLU()(out)
        out = self.fc(out[:, -1, :])

        return out, {}