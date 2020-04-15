from torchvision import models
import torch.nn as nn


def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes, pretrained = False):
    ## your code here
    pass


def model_C(num_classes, pretrained = False):
    ## your code here
    pass