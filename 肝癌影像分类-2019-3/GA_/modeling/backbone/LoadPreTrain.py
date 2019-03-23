# -*- coding: utf-8 -*-
"""
@author: 赵磊
@project: UGPU_Server
@file: LoadPreTrain.py
@time: 2019/3/8 10:06
@description:
"""

from modeling.backbone import resnet, resnetv2, resnext
from modeling.img_classifier import *
import torch.utils.model_zoo as model_zoo
import pdb
import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}


class PreTrainLoader():
    def __init__(self):
        self.model = Classifier(num_classes=2,
                                backbone='resnet',
                                output_stride=16,
                                sync_bn=None,
                                freeze_bn=False)
        self.model=self.model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

if __name__ == '__main__':
    model_mine = Classifier(num_classes=2,
                            backbone='resnet',
                            output_stride=16,
                            sync_bn=None,
                            freeze_bn=False)

    sta_model_res101=models.resnet101(pretrained=True)

    pretrained_dict=sta_model_res101.state_dict()
    model_dict = model_mine.state_dict()

    print(pretrained_dict.items())
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict_tmp = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model_mine.load_state_dict(model_dict)

    print(sta_model_res101)
    print(model_mine)
