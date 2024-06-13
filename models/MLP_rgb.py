from torch import nn

class MLP_aggregation_classifier(nn.Module):

    #subsample and num layer are parameters added just for consistency with the other models
    #to use just a single train file
    def __init__(self, num_classes=8, input_size=1024, hidden_size=512, dropout_prob=0.5,subsample_num=3, num_layers=None):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # AdaptiveAvgPool1d for averaging along the sequence dimension

    def forward(self, x):

        x = self.avg_pool(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.fc1(x.squeeze(dim=1))
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x, {}
