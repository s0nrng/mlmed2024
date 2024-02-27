import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

X_train = pd.read_csv("mitbih_train.csv").to_numpy()
X_test = pd.read_csv("mitbih_test.csv").to_numpy()

Y_train = X_train[67000:,-1].astype(int)
Y_test = X_test[:,-1].astype(int)
X_train = X_train[67000:,:-1]
X_test = X_test[:,:-1]

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(32, 32, 5, padding='same')
        init.xavier_uniform_(self.conv1.weight)
        init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv1d(32, 32, 5, padding='same')
        init.xavier_uniform_(self.conv2.weight)
        init.constant_(self.conv2.bias, 0)
        self.pool = nn.MaxPool1d(5,stride=2)
        
    def forward(self, data):
        output = f.relu(self.conv1(data))
        output = self.conv2(output)
        output = output + data
        output = f.relu(output)
        output = self.pool(output)
        return output
    
class implementedModel(nn.Module):
    def __init__(self, device):
        super(implementedModel, self).__init__()
        self.preconv = nn.Conv1d(1, 32, 5, padding='same')
        init.xavier_uniform_(self.preconv.weight)
        init.constant_(self.preconv.bias, 0)
        self.res = []
        for i in range(5):
            self.res.append(ResBlock().to(device))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        
        
    def forward(self, data):
        data = self.preconv(data)
        for resblock in self.res:
            data = resblock(data)
        data = nn.Flatten()(data)
        data = f.relu(self.fc1(data))
        data = f.softmax(self.fc2(data))
        return data

class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X).unsqueeze(dim=1)
        self.Y = torch.LongTensor(Y)
        
    def __len__(self):
        return(self.Y.shape[-1])
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_dataset = EEGDataset(X_train, Y_train)
test_dataset = EEGDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64)

device = "mps"
model = implementedModel(device).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

epoch = 100
train_hist = []
test_hist = []
for e in range(epoch):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (datas, labels) in enumerate(tqdm(train_loader)):
        datas = datas.to("mps")
        labels = labels.to("mps")
        optimizer.zero_grad()
        outputs = model(datas)
        loss = loss_fn(outputs, labels)
        loss.backward()
        train_loss += loss.item()
        train_acc += (np.sum(np.array(outputs.argmax(dim=1).tolist()) == np.array(labels.tolist())))
        optimizer.step()
    train_loss /= len(train_loader)
    train_acc /= X_train.shape[0]
    
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for i, (datas, labels) in enumerate(tqdm(test_loader)):
            datas = datas.to("mps")
            labels = labels.to("mps")
            outputs = model(datas)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            test_acc += (np.sum(np.array(outputs.argmax(dim=1).tolist()) == np.array(labels.tolist())))
        test_loss /= len(test_loader)
        test_acc /= X_test.shape[0]
        
    train_hist.append(train_loss)
    test_hist.append(test_loss)
    print("Epoch: ",e," | Train Accuracy: ", train_acc," | Test Accuracy:", test_acc)

