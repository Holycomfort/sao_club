from torchvision import models
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def model_A(num_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet

def model_B(num_classes, pretrained = False):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, num_classes)
    return model_resnet


def model_C(num_classes, pretrained = False):
    model = Network()
    return model


class Network(nn.Module):
    def __init__(self, mode='stn'):
        assert mode in ['stn', 'cnn']

        super(Network, self).__init__()
        self.mode = mode
        self.local_net = LocalNetwork()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=4, bias=False),
            nn.BatchNorm2d(16)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=224 // 4 * 224 // 4 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=20)
        )

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size = img.size(0)
        if self.mode == 'stn':
            transform_img = self.local_net(img)
            img = transform_img
        else:
            transform_img = None

        conv_output = nn.ReLU()(self.conv(img)+self.shortcut(img)).view(batch_size, -1)
        predict = self.fc(conv_output)
        return transform_img, predict


class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 224 * 224,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, 3, 224, 224)))
        img_transform = F.grid_sample(img, grid)

        return img_transform


if __name__ == "__main__":
    pass