from torchvision import models
from torch import nn
from setting import *


def get_model():
    # 定义模型
    _model = models.resnet34(pretrained=True)
    _model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return _model
