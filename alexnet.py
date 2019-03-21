import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as image
import torch
import torchvision
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional as tf

__all__ = ['AlexNet', 'alexnet']
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
}

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 55 x 55
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 27 x 27

            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # 27 x 27
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.view = lambda x: x.view(x.size(0), 256 * 6 * 6)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),

            nn.Linear(4096, num_classes),
        )
        self.softmax = torch.nn.Softmax()

        self.layers = [
            self.features[0],
            self.features[1],
            self.features[2],
            self.features[3],
            self.features[4],
            self.features[5],
            self.features[6],
            self.features[7],
            self.features[8],
            self.features[9],
            self.features[10],
            self.features[11],
            self.features[12],
            self.avgpool,
            self.view,

            self.classifier[0],     #15
            self.classifier[1],     #16
            self.classifier[2],     #17
            self.classifier[3],     #18
            self.classifier[4],     #19
            self.classifier[5],     #20
            self.classifier[6],     #21
            self.softmax            #22
        ]
        self.n_layers = len(self.layers)

    def forward(self, x, start=0, end=-1):
        if end < 0:
            end = self.n_layers

        for i in range(start, end):
            x = self.layers[i](x)

        return x

def alexnet(pretrained=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    return model
