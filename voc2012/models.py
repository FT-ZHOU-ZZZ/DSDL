import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn


class DSDL(nn.Module):
    def __init__(self, model, num_classes, alpha, in_channel=300):
        super(DSDL, self).__init__()
        self.alpha = alpha

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.W1 = nn.Parameter(torch.zeros(size=(in_channel, 1024)))
        stdv = 1. / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv, stdv)
        self.relu = nn.LeakyReLU(0.2)
        self.W2 = nn.Parameter(torch.zeros(size=(1024, 2048)))
        stdv = 1. / math.sqrt(self.W2.size(1))
        self.W2.data.uniform_(-stdv, stdv)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        semantic = torch.matmul(inp[0], self.W1)
        semantic = self.relu(semantic)
        semantic = torch.matmul(semantic, self.W2)

        res_semantic = torch.matmul(semantic, self.W2.transpose(0, 1))
        res_semantic = self.relu(res_semantic)
        res_semantic = torch.matmul(res_semantic, self.W1.transpose(0, 1))
        score = torch.matmul(
            torch.inverse(torch.matmul(semantic, semantic.transpose(0, 1)) + self.alpha * torch.eye(self.num_classes).cuda()),
            torch.matmul(semantic, feature.transpose(0, 1))).transpose(0, 1)

        return score, inp[0], res_semantic, feature, semantic

    def get_config_optim(self, lr, lrp):
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': self.W1, 'lr': lr},
            {'params': self.W2, 'lr': lr},
        ]


def load_model(num_classes, alpha, pretrained=True, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return DSDL(model, num_classes, alpha, in_channel=in_channel)
