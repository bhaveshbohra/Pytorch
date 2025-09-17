## Import necessary lib

import torch 
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
#from torchvision.models import vgg16
import torchvision

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# hyperparameter 
in_channel= 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epoch = 10
# load_ckpt = True ## load_model = True (rename)
# ckpt_file = "my_checkpoint.pyh.tar"


# load pretrain model and modify it 
# Torch-vision model 
# import sys 

## 01 Original 
# model = torchvision.models.vgg16(pretrained = True)
# print(model)
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (1): ReLU(inplace=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)        
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)        



# 02 Modified avg pool and classifier with single layer 
# modifed the classifier for output feature 10
# removing avgpool (7,7) 

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self,x):
    return x 
  
model = torchvision.models.vgg16(pretrained = True)
model.avgpool = Identity()
model.classifier = nn.Linear(512,10)
model.to(device)

# print(model)
# sys.exit()
#   (avgpool): Identity()
#   (classifier): Linear(in_features=512, out_features=10, bias=True)    


## 03 Modified specific layer like layer for specific layer[0][1]
# model.classifier[0] = nn.Linear(512,10) 
# print(model)

#   (avgpool): Identity()
#   (classifier): Sequential(
#     (0): Linear(in_features=512, out_features=10, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)        
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)        


## 04 modified with multiple layer using sequential 
# model.classifier = nn.Sequential(nn.Linear(512,100),
#                                  nn.Dropout(p=0.3),
#                                  nn.Linear(100,10))
# print(model)

#   (avgpool): Identity()
#   (classifier): Sequential(
#     (0): Linear(in_features=512, out_features=100, bias=True)
#     (1): Dropout(p=0.3, inplace=False)
#     (2): Linear(in_features=100, out_features=10, bias=True)

# def save_checkpoint(state, filename ="my_checkpoint.pth.tar"):
#     print("=> Saving Checkpoint")
#     torch.save(state, filename)
  
# def load_checkpoint(checkpoint):
#   print("=> loading Checkpoint")
#   model.load_state_dict(checkpoint['state_dict'])
#   optimizer.load_state_dict(checkpoint['optimizer'])
#   ## add epoch.  accuracy 


# load Dataset 
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

## initialize Network 
# model = CNN().to(device) # CNN(all are default we set)

## loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# import os 
# if load_ckpt and os.path.isfile(ckpt_file):
#   load_checkpoint(torch.load(ckpt_file, map_location =device))

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
# # Save checkpoint every 3 epochs
#   if epoch % 3 ==0: # means after 3 epoch model save 
#     checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
#     save_checkpoint(save_checkpoint)

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
# check_accuracy(test_loader, model)

