from torch import nn
from utils.logger import logger



class EMG_LSTM(nn.Module):
    
    def __init__(self, num_classes=20):
        super().__init__()      
        self.lstm = nn.LSTM(input_size=16, hidden_size=50, batch_first=True)
  
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=50, out_features=num_classes)
   
    def forward(self, x):
            
        x = x.float()
        x=x.squeeze(dim=1)
        out, _ = self.lstm(x)
        mid_level_features = {}
        mid_level_features['features'] = out[:, -1, :]

        out = self.dropout(out)
        out= self.relu(out)

        out = self.fc(out[:, -1,:])
        return out, mid_level_features