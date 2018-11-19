import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
nclasses = 20 

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.model = models.resnet152(pretrained=True)
    num_features = self.model.fc.in_features
    self.model.fc = nn.Linear(num_features, nclasses)
  def forward(self, x):
    x = self.model.forward(x)
    return x

