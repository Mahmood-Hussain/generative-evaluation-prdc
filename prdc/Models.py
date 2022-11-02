import torch.nn as nn
from torchvision.models import vgg16

class RandomX(nn.Module):
    def __init__(self, out_feats=64):
        super(RandomX, self).__init__()
        self.vgg16 = vgg16(pretrained=False)
        self.features = self.vgg16.features.requires_grad_(False)
        self.fc1 = nn.Linear(25088, out_feats).requires_grad_(False)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class TrainedX(nn.Module):
    def __init__(self, out_feats=64):
        super(TrainedX, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.features = self.vgg16.features.requires_grad_(False)
        self.fc1 = self.vgg16.classifier[0].requires_grad_(False)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x