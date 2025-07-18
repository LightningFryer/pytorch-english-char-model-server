data_dir = 'C:/Users/User/.cache/kagglehub/datasets/mohneesh7/english-alphabets/versions/1/english_alphabets'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_functional
from torchvision.datasets import ImageFolder
import timm

class EnglishCharacterDataset(Dataset):
  def __init__(self, data_dir, transform = None):
    self.data = ImageFolder(data_dir, transform=transform)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

  @property
  def classes(self):
    return self.data.classes
  
from random import randint

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = EnglishCharacterDataset(data_dir=data_dir, transform=transform)

# Test to see if dataset loads
print("Length of dataset: ", len(dataset))
image, label = dataset[randint(0, len(dataset) - 1)]
image

for image, label in dataset:
    break

# Making a dataloader in order to create batches to train
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

# Testing by iterating and breaking, we'll be using images as the testing in the model down below
for images, labels in dataloader:
  break

# Creating the classifier (untrained base model) to train from timm, the model class is called a 'Classifier' dunno why
class EnglishCharacterClassifier(nn.Module):
  def __init__(self, numClasses=26): # numClasses here is 26 because of 26 letters in the english alphabet
    super(EnglishCharacterClassifier, self).__init__()
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    enet_out_size = 1280
    # Make a classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(enet_out_size, numClasses)
    )

  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)

    return output

# Creating a new model from the classifier above
model = EnglishCharacterClassifier(numClasses=26)
# print(model)

# Testing the model by passing one 'images' batch from the dataloader test from above
model(images) # Should output a tensor if works
output = model(images)
print(output.shape)

for images, labels in dataloader:
  continue

# TRAINING THE MODEL STARTS HERE
# First we need to make 2 things: a loss function (called a criterion) and an optimizer
model = EnglishCharacterClassifier(numClasses=26)
model(images)
output = model(images)
print(output.shape)

criterion = nn.CrossEntropyLoss();
optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion(output, labels)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])