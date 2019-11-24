# -*- coding: utf-8 -*-
# @Time    : 19-10-21 上午9:11
# @Author  : Zhao Lei
# @File    : model_without_ASPP_resnet101.py
# @Desc    :

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo


################################################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        self._init_weight()
        if layers == [3, 4, 23, 3]:
            self._load_pretrained_model2()
        else:
            self._load_pretrained_model1()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        feat1 = x
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = x
        x = self.layer2(x)
        feat3 = x
        x = self.layer3(x)
        feat4 = x
        x = self.layer4(x)
        feat5 = x
        return feat1, feat2, feat3, feat4, feat5

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model1(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print('backbone pretrained model 50 loaded~')

    def _load_pretrained_model2(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        print('backbone pretrained model 101 loaded~')


def ResNet50(output_stride, BatchNorm):
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm)
    return model


def ResNet101(output_stride, BatchNorm):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm)
    return model


class JPU(nn.Module):
    def __init__(self, width=512, norm_layer=None, ):
        super(JPU, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(2048, width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.postconv = nn.Conv2d(1536, 256, kernel_size=3, stride=1, padding=1)

        self._init_weight()

    def forward(self, feat3, feat4, feat5):
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)
        feat5 = self.conv5(feat5)

        feat4 = F.interpolate(feat4, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat5 = F.interpolate(feat5, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat = torch.cat((feat5, feat4, feat3), dim=1)

        f_feat = self.postconv(feat)

        return f_feat

    def _init_weight(self):
        print('start JPU Module Initialization~')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(Decoder, self).__init__()

        low_level_inplanes = 256
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat4x):
        low_level_feat4x = self.conv1(low_level_feat4x)
        low_level_feat4x = self.bn1(low_level_feat4x)
        low_level_feat4x = self.relu(low_level_feat4x)
        x = F.interpolate(x, size=low_level_feat4x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat4x), dim=1)
        x = self.last_conv1(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm):
    return Decoder(num_classes, BatchNorm)


class DeepLab_Res_JPU(nn.Module):
    def __init__(self, output_stride, num_classes,
                 ):
        super(DeepLab_Res_JPU, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = ResNet101(output_stride=output_stride, BatchNorm=BatchNorm)
        self.jpu = JPU(norm_layer=BatchNorm)
        self.decoder = build_decoder(num_classes=num_classes, BatchNorm=BatchNorm)

    def forward(self, input):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(input)
        x = self.jpu(feat3, feat4, feat5)
        x = self.decoder(x, feat2)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
