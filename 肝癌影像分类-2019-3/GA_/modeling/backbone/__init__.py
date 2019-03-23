# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午10:53
# @Author  : Zhao Lei
# @File    : __init__.py.py
# @Desc    :

from modeling.backbone import resnet,resnext,resnetv2


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.resnet101(pretrained=False)
    elif backbone == 'resnetv2':
        return resnetv2.resnet101(pretrained=True)
    elif backbone == 'resnext':
        return resnext.resnext101(pretrained=True)
    else:
        raise NotImplementedError
