# -*- coding: utf-8 -*-
# @Time    : 19-10-16 下午12:14
# @Author  : Zhao Lei
# @File    : InferKernel.py
# @Desc    :

#####################################model definition#######################################
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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

        self._init_weight()

    def forward(self, feat3, feat4, feat5):
        feat3 = self.conv3(feat3)
        feat4 = self.conv4(feat4)
        feat5 = self.conv5(feat5)

        feat4 = F.interpolate(feat4, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat5 = F.interpolate(feat5, size=feat3.size()[2:], mode='bilinear', align_corners=True)
        feat = torch.cat((feat5, feat4, feat3), dim=1)

        return feat

    def _init_weight(self):
        print('start JPU Module Initialization~')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        inplanes = 512
        dilations = [1, 2, 4, 8]
        self.preconv = nn.Conv2d(1536, 512, kernel_size=3, stride=1, padding=1)
        self.aspp1 = _ASPPModule(inplanes, 256, 3, padding=1, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.conv1 = nn.Conv2d(1024, 256, 1, bias=False)

    def forward(self, x):
        x = self.preconv(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(output_stride, BatchNorm):
    return ASPP(output_stride, BatchNorm)


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
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
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

        self.backbone = ResNet50(output_stride=output_stride, BatchNorm=BatchNorm)
        self.jpu = JPU(norm_layer=BatchNorm)
        self.aspp = build_aspp(output_stride=output_stride, BatchNorm=BatchNorm)
        self.decoder = build_decoder(num_classes=num_classes, BatchNorm=BatchNorm)

    def forward(self, input):
        feat1, feat2, feat3, feat4, feat5 = self.backbone(input)
        x = self.jpu(feat3, feat4, feat5)
        x = self.aspp(x)
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
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


######################################test images and submission.csv#############################################
sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'
test_data_folder = "../input/severstal-steel-defect-detection/test_images"


##########################################data reading###########################################
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        img = np.array(image).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        img = np.array(image).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img


class TestDataset(Dataset):
    '''Dataset for test prediction'''

    def __init__(self, root, df):
        self.root = root
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform_te(image)
        return fname, images

    def __len__(self):
        return self.num_samples

    def transform_te(self, sample):
        composed_transforms = transforms.Compose([
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])
        return composed_transforms(sample)


def mask2c_maskf(mask2c):
    cat1 = np.zeros((256, 1600), dtype=np.float32)
    cat2 = np.zeros((256, 1600), dtype=np.float32)
    cat3 = np.zeros((256, 1600), dtype=np.float32)
    cat4 = np.zeros((256, 1600), dtype=np.float32)
    cat1[mask2c == 1] = 1.0
    cat2[mask2c == 2] = 1.0
    cat3[mask2c == 3] = 1.0
    cat4[mask2c == 4] = 1.0
    ### 这里可能展开的时候需要order='F',因为默认是按行的
    return [1, 2, 3, 4], [cat1.reshape(256 * 1600,order='F'), cat2.reshape(256 * 1600,order='F'), cat3.reshape(256 * 1600,order='F'),
                          cat4.reshape(256 * 1600,order='F')]


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


########################################################################################

# initialize test dataloader
num_workers = 5
batch_size = 3
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

##########################################################################################################
##训练集2测试集1的比例
## epoch16 之前
# ckpt_path = "../input/epoch16pre/model_best.pth.tar"## V2-当前阶段miou最高
# ckpt_path = "../input/epoch16pre/checkpoint_epoch14.pth.tar"##V3-当前阶段val 损失最低
## epoch64 之前
ckpt_path = "../input/epoch64pre/checkpoint_epoch63.pth.tar"##V4-当前阶段最新,pnglabel模式下最佳
##########################################################################################################
##训练集9验证集1的比例
# ckpt_path = "../input/tvraepoch5pre/model_best.pth.tar"## V5-当前阶段miou最高
ckpt_path = "../input/tvraepoch5pre/checkpoint_epoch5.pth.tar"## V6-当前阶段最新
##########################################################################################################
device = torch.device("cuda")
model = DeepLab_Res_JPU(output_stride=16, num_classes=5)
model.to(device)
model.eval()
state_dict_toload = torch.load(ckpt_path)
model.load_state_dict(state_dict_toload.get('state_dict'))
print("trained model loaded!")

# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    outputs = model(images.to(device))
    preds = outputs.data.cpu().numpy()
    batch_preds = np.argmax(preds, axis=1)

    for fname, pred in zip(fnames, batch_preds):
        clss, locarrs = mask2c_maskf(pred)
        for cls, locarr in zip(clss, locarrs):
            rle = mask2rle(locarr)
            name = fname + f"_{cls}"
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
df.to_csv("submission.csv", index=False)
