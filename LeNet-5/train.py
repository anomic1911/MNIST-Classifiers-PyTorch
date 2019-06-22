from model import LeNet
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("==>>>device is", device)
# Hyperparams
num_classes = 10
num_epochs = 20
lr = 1e-3
batch_size = 100 

root = './../../datasets/mnist'
if not os.path.exists(root):
    os.mkdir(root)

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

print("==>>> total number of training batches : {}".format(len(train_loader)))
print("==>>> total number of testing batches : {}".format(len(test_loader)))
print("==>>> Batch Size is : {}".format(batch_size))


model = LeNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

num_batches = len(train_loader)

for epoch in range(num_epochs):
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((idx+1)%100==0):
            print("epoch is {}/{} Step is: {}/{} loss is: {}".format(epoch, num_epochs, idx, num_batches, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        values, indices = torch.max(preds, 1)
        total += labels.shape[0] 
        correct += (labels == indices).sum().item()
    print("Accuracy of the network is: {}%".format(100*correct / total) )

torch.save(model.state_dict(), 'model.pth')        
