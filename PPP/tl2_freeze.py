## Import necessary lib

import torch 
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torchvision.models import vgg16, VGG16_Weights
import torchvision

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# hyperparameter 
in_channel= 3
num_classes = 10
learning_rate = 0.001
batch_size = 1024
num_epoch = 2

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self,x):
    return x 


# load pretrained model and modify it 
# model = torchvision.models.vgg16(pretrained = True)
model = vgg16(weights=VGG16_Weights.DEFAULT)

## freeze layer 
for param in model.parameters():
  param.requires_grad = False   
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),
                                 nn.Dropout(p=0.3),
                                 nn.Linear(100,10))
print(model)
model.to(device)


# load Dataset 
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


## loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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



## new way or new version of pytorch 

# Instead of:

# model = torchvision.models.vgg16(pretrained=True)

# use below 

# from torchvision.models import vgg16, VGG16_Weights

# weights = VGG16_Weights.DEFAULT
# model = vgg16(weights=weights)

# from torchvision.models import vgg16 , VGG16_Weights
# model = vgg16(weights=VGG16_Weights.DEFAULT)


# document 

# torchvision.models.vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) → VGG
# Breakdown:
# vgg16(...): The function you call to create/load a VGG-16 model.

# weights: Optional[VGG16_Weights] = None:

# This replaces the old pretrained=True.

# If you want pretrained weights on ImageNet:

# python
# Copy
# Edit
# from torchvision.models import VGG16_Weights
# weights = VGG16_Weights.DEFAULT
# model = vgg16(weights=weights)
# If you want a randomly initialized model (no pretrained):

# python
# Copy
# Edit
# model = vgg16(weights=None)
# progress: bool = True:

# If True, shows a download progress bar when downloading pretrained weights.

# Usually just leave it as True.

# **kwargs: Any:

# Accepts any other optional keyword arguments like num_classes=..., etc.

# → VGG:

# The function returns a VGG model (i.e., a PyTorch nn.Module subclass).



