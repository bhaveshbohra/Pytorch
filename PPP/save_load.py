## Import necessary lib

import torch 
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  


# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# hyperparameter 
in_channel= 1
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epoch = 10
load_ckpt = True ## load_model = True (rename)
ckpt_file = "my_checkpoint.pyh.tar"

## create simple CNN

class CNN(nn.Module):
  def __init__(self, in_channels=1 , num_classes =10):
    super(CNN, self).__init__()
    self.conv1= nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv2= nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.fc1 = nn.Linear(8*7*7, num_classes)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)

    return x
  
def save_checkpoint(state, filename ="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
  
def load_checkpoint(checkpoint):
  print("=> loading Checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  ## add epoch.  accuracy 


# load Dataset 
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

## initialize Network 
model = CNN().to(device) # CNN(all are default we set)

## loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import os 
if load_ckpt and os.path.isfile(ckpt_file):
  load_checkpoint(torch.load(ckpt_file, map_location =device))

## replced with load_ckpt
# if load_model:
#   load_model(torch.load("my_checkpoint.pth.tar"))

## Train Network
for epoch in range(num_epoch):
  losses =[]
  for batch_idx, (data, targets) in enumerate(train_loader):
    # Get data to cuda if possible
    data = data.to(device=device)
    targets = targets.to(device=device)

    # forward
    scores = model(data)
    loss = criterion(scores, targets)
    losses.append(loss.item())  # ✅ collect loss

    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient descent or adam step
    optimizer.step()

  avg_loss = sum(losses) / len(losses)
# Save checkpoint every 3 epochs
  if epoch % 3 ==0: # means after 3 epoch model save 
    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(save_checkpoint)

  print(f"Loss at epoch {epoch} was {avg_loss:.4f}")

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
    #   x = x.reshape(x.shape[0], -1)

      scores = model(x)

      _,predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)

    print(f'Get {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

  model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

