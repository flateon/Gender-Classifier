from torch import nn
from torchvision import models


def get_model():
    # 定义模型
    model_ = models.resnet34(pretrained=True)
    model_.fc = nn.Linear(in_features=512, out_features=1, bias=True)
    return model_
