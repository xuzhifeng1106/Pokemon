import torch
import torchvision
from torch import nn

# 加载预训练的 ResNet18 模型
# model_resnet = torchvision.models.resnet18(pretrained=True)
#
# model_resnet.fc = nn.Linear(num_features, 5)


class PokemonClassifier(nn.Module):
    def __init__(self):
        super(PokemonClassifier, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 5)

    def forward(self, x):
        x = self.model(x)
        return x
