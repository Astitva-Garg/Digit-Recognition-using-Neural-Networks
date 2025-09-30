import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    num_epochs = 5
    batchsize = 10
    learning_rate = 0.001
    train_data = pd.read_csv("Data/train.csv")
    
    Y_train = train_data.iloc[:35000, 0].values # training data
    X_train = train_data.iloc[:35000, 1:].values
    X_train = torch.tensor(X_train, dtype=torch.float32) / 255.0   # normalize 
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(dataset = train_dataset, batch_size= batchsize, shuffle= True)
    
    Y_test = train_data.iloc[35000:, 0].values # test data
    X_test = train_data.iloc[35000:, 1:].values
    X_test = torch.tensor(X_test, dtype=torch.float32) / 255.0   
    Y_test = torch.tensor(Y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, Y_test)
    test_loader = DataLoader(dataset = test_dataset, batch_size= batchsize, shuffle= False)
    
    num_input_features = 784
    num_output_features = 100
    num_classes = 10

    model = NeuralNetwork(num_input_features, num_output_features, num_classes).to(device)
    CSEloss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    
    n_total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = CSEloss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_step}, loss: {loss.item():.4f}")

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 7000 test images: {acc} %')


    
class NeuralNetwork(nn.Module):
    def __init__(self, num_input_features, num_output_features, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(num_input_features, num_output_features)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(num_output_features, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

main()








