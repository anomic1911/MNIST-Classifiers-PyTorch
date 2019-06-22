import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms, datasets
from model import NeuralNet

root = './../data/'
if not os.path.exists(root):
    os.mkdir(root)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("==>>> device is", device)

# Hyper-parameters 
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 20
lr = 0.001
batch_size = 100

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])   

# mean : It is the mean of all pixel values in the dataset ( 60000 × 28 × 28 ).
# This mean is calculated over the whole dataset.
# deviation : It is the standard deviation of all pixel values.
# The dataset is treated as a population rather than a sample.

train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)
print("==>>> total trainning batch number: {}".format(len(train_loader)))
print("==>>> total testing batch number: {}".format(len(test_loader)))
print("==>>> total number of batches are: {}".format(batch_size))

for index, batch in enumerate(train_loader):
    inputs = batch[0]
    labels = batch[1]
    if(index == 0):
        print("==>>> input shape of a batch is: {}".format(inputs.shape))
        print("==>>> labels shape of a batch is: {}".format(labels.shape))
        

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

num_batches = len(train_loader)
train_loss = []
epoch_counter = []
cnt = 0
for epoch in range(num_epochs):
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.reshape(-1,28*28).to(device)
        labels = labels.to(device)
        # Forward propagation        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # backward pass and make step         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        cnt += 1
        epoch_counter.append(cnt)
        if((idx+1)%100==0):
            print("epoch is {}/{} Step is: {}/{} loss is: {}".format(epoch, num_epochs, idx, num_batches, loss.item()))
plt.plot(epoch_counter, train_loss)
torch.save(model.state_dict(), 'model.pth')        
torch.save(model.state_dict(), 'optimizer.pth')    
with torch.no_grad():
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        preds = model(inputs)
        values, indices = torch.max(preds, 1)
        total += labels.shape[0] 
        correct += (labels == indices).sum().item()
    print("Accuracy of the network is: {}%".format(100*correct / total) )
