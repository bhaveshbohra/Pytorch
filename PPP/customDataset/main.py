## Import necessary lib

import torch 
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F  
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torchvision.models import googlenet, GoogLeNet_Weights
import torchvision
from load_customdata import CatsAndDogsDataset

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# hyperparameter 
in_channel= 3
num_classes = 2
learning_rate = 0.0001
batch_size = 32
num_epoch = 10

## Load custom_data 
dataset= CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform = transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [5, 5]) # actually 20000,5000
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# load pretrained model and modify it 
# model = torchvision.models.vgg16(pretrained = True)
model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
model.to(device)
# ## freeze layer 
# for param in model.parameters():
#   param.requires_grad = False   
# model.avgpool = Identity()
# model.classifier = nn.Sequential(nn.Linear(512,100),
#                                  nn.Dropout(p=0.3),
#                                  nn.Linear(100,10))
# print(model)
# model.to(device)


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
    losses.append(loss.item())  # collect loss

    # backward
    optimizer.zero_grad()
    loss.backward()

    # gradient descent or adam step
    optimizer.step()

  avg_loss = sum(losses) / len(losses)
  print(f"Loss at epoch {epoch} was {avg_loss:.4f}")

# check accuracy on training & test to see how good or not 
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)