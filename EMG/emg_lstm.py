from torch import nn


class EMG_LSTM(nn.Module):
    
   
    def __init__(self):
        super().__init__()
       
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=50, batch_first=True)
       
        #self.lstm2 = nn.LSTM(input_size=5, hidden_size=50, batch_first=True)
       
        self.dropout = nn.Dropout(p=0.2)
     
        self.fc = nn.Linear(in_features=50, out_features=21)
        
    def forward(self, x):
        
        out1, _ = self.lstm(x)
        
        out2, _ = self.lstm2(out1)
       
        out_last = out2[-1]
      
        out_dropout = self.dropout(out_last)
        
        out = self.fc(out_dropout)
        
        return out