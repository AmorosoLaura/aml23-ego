import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from emg_lstm import EMG_LSTM
from emg_loader import EMG_dataset


if __name__ == '__main__':

    train_set = EMG_dataset('./', 'emg_data_preprocessed_train.pkl')
                            
    test_set = EMG_dataset('./', 'emg_data_preprocessed_train.pkl')
    train_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=2)

    test_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=2)
    # Initialize the model, loss function, and optimizer
    model = EMG_LSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Initialize lists to store top 1 and top 5 accuracies
    top1_accuracy_list = []
    top5_accuracy_list = []

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):

        model.train()
        print(epoch)
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # Get top 5 predictions
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in predicted_top5[i]])
        
        # Calculate accuracy
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_top1_accuracy = correct_top1 / total
        epoch_top5_accuracy = correct_top5 / total
        
        # Store top 1 and top 5 accuracies
        top1_accuracy_list.append(epoch_top1_accuracy)
        top5_accuracy_list.append(epoch_top5_accuracy)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Top 1 Accuracy: {epoch_top1_accuracy:.4f}, Top 5 Accuracy: {epoch_top5_accuracy:.4f}")

    # Print top 1 and top 5 accuracies over the epochs
    print("Top 1 Accuracy over epochs:", top1_accuracy_list)
    print("Top 5 Accuracy over epochs:", top5_accuracy_list)


    # Initialize lists to store top 1 and top 5 accuracies for test data
    test_top1_accuracy_list = []
    test_top5_accuracy_list = []

    # Test the model
    for epoch in range(num_epochs):
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
                
                # Get top 5 predictions
                _, predicted_top5 = torch.topk(outputs, 5, dim=1)
                correct_top5 += sum([1 for i in range(len(labels)) if labels[i] in predicted_top5[i]])
        
        # Calculate accuracy
        test_top1_accuracy = correct_top1 / total
        test_top5_accuracy = correct_top5 / total
        
        # Store top 1 and top 5 accuracies for test data
        test_top1_accuracy_list.append(test_top1_accuracy)
        test_top5_accuracy_list.append(test_top5_accuracy)
        
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Top 1 Accuracy: {test_top1_accuracy:.4f}, Test Top 5 Accuracy: {test_top5_accuracy:.4f}")

    # Print top 1 and top 5 accuracies for test data over the epochs
    print("Test Top 1 Accuracy over epochs:", test_top1_accuracy_list)
    print("Test Top 5 Accuracy over epochs:", test_top5_accuracy_list)