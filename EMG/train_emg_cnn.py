import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from emg_lenet import LeNet5

num_classes = 20  
model = LeNet5(num_classes).float()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


class SpectroDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram = self.data[idx]['spectrogram']
        label = self.data[idx]['label']
        #print(spectrogram.shape)
        label = torch.tensor(label, dtype=torch.long)
        label = torch.squeeze(label)
        return spectrogram, label

train_file = './EMG_data/emg_spectrogram_train.pkl'
test_file = './EMG_data/emg_spectrogram_test.pkl'

train_dataset = SpectroDataset(train_file)
test_dataset = SpectroDataset(test_file)


batch_size = 10
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# Training loop
epochs = 10

print("TRAIN")
model.train()  # Set the model to training mode
for epoch in range(epochs):
    print(epoch)
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
       
        outputs = model(inputs)  # Forward pass
        
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

""" print("TEST")
# Validation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {accuracy:.4f}")

"""
