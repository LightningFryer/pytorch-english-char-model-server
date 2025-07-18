
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

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