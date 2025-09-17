## Import necessary lib

import torch ## create tensor,  move into gpu, device setup
import torch.nn as nn ## build block
import torch.optim as optim ## optimzer 
import torch.nn.functional as F ## activation function 
from torch.utils.data import DataLoader ## helper function for load dataset 
import torchvision.datasets as datasets ## Ready‑made dataset classes
import torchvision.transforms as transforms  ##Convert PIL/NumPy data into tensors, pre‑processing

## Create Fully connected Network 

class NN(nn.Module): ## create NN class from(inheritance) nn module 
  def __init__(self, input_size, num_classes):  ## initialize ,  in over case input size 784(28*28)
    super(NN, self).__init__() ## super is called initialization
    self.fc1 = nn.Linear(input_size, 50) ## so we put inputsize , num classes , hidden layer is 50 node 
    self.fc2 = nn.Linear(50, num_classes) ## hidden layer 50, output 

  def forward(self, x):
    x = F.relu(self.fc1(x)) ## activation function 
    x = self.fc2(x)
    return x

# model = NN(784,10)
# x = torch.randn(64,784)
# print(model(x).shape)

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# hyperparameter 
input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epoch = 5


# load Dataset 
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

## initialize Network 
model = NN(input_size=input_size, num_classes=num_classes).to(device)


## loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

## Train Network
for epoch in range(num_epoch):
  for batch_idx, (data, targets) in enumerate(train_loader):
    # Get data to cuda if possible
    data = data.to(device=device)
    targets = targets.to(device=device)

    # print(data.shape)
    # Get to correct shape
    data = data.reshape(data.shape[0], -1)

    # print(data.shape)

    # forward
    scores = model(data)
    loss = criterion(scores, targets)

    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient descent or adam step
    optimizer.step()

# check accuracy on training & test to see how good or not 

def check_accuracy(loader, model):

  if loader.dataset.train:
    print("Checking accuracy on training data")
  else:
    print("Checking accuracy on test data")

  num_correct = 0 
  num_samples = 0 
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.reshape(x.shape[0], -1)

      scores = model(x)

      _,predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

    print(f'Get {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

  model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

