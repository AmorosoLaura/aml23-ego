from torch import nn


class EMG_LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
       
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=50, batch_first=True)
       
        #self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, batch_first=True)
       
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=50, out_features=20)
   
    def forward(self, x):
            
        x = x.float()
        #x=x.squeeze(dim=1)

        out, _ = self.lstm(x)
        #out2, _ = self.lstm2(out1)
       
        #out_last = out1[-1]
      
        out = self.dropout(out)
        out= self.relu(out)

        out = self.fc(out[:, -1,:])
        return out, {}