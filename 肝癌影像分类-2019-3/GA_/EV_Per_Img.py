# -*- coding: utf-8 -*-
# @Time    : 19-3-14 下午7:55
# @Author  : Zhao Lei
# @File    : EV_Per_Img.py.py
# @Desc    :


import pdb
import random
from tqdm import tqdm

from modeling.img_classifier import *
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import copy
from torchvision import transforms
from dataloaders import c_t_test as te


def transform_te(sample):
    composed_transforms = transforms.Compose([
        te.FixScaleCrop(crop_size=513),
        te.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        te.ToTensor()])

    return composed_transforms(sample)


model = Classifier(num_classes=2,
                   backbone='resnetv2',
                   output_stride=16,
                   sync_bn=None,
                   freeze_bn=False)
state_dict_toload = torch.load(
    '/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/experiment_resnetv2_100_pretrained/checkpoint.pth.tar')
model.load_state_dict(state_dict_toload.get('state_dict'))
model = torch.nn.DataParallel(model, device_ids=[0])
model = model.cuda()
model.eval()

# path = "test/"
path = "/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/train_dataset1_jpg/"
dirlist = []  # all dir name
for root, dirs, files in os.walk(path):
    for dir in dirs:
        dirlist.append(dir)
v_count = 0
for dir in dirlist:
    # images in each folder
    v_count = v_count + 1
    print(v_count)
    file_a = open(path + dir + '.txt', 'a+')
    for root, dirs, files in os.walk(path + dir):  # traverse each folder
        print(path + dir + '')
        mergeArr_list_tU = []
        for file in files:
            pilImage = Image.open(path + dir + '/' + file)
            img_arr = np.array(pilImage)
            mergeArr_list_tU.append(img_arr)
        tU_arr_all = np.zeros((512, 512, 3))
        for ii_a in range(len(files)):
            tU_arr_all = tU_arr_all + mergeArr_list_tU[ii_a]
        tU_arr_all = tU_arr_all / len(mergeArr_list_tU)
        tU_pil = Image.fromarray(tU_arr_all.astype(np.uint8)).convert('RGB')
        sample = {'image': tU_pil}
        sample = transform_te(sample)
        image = sample['image']
        image = image.reshape((1, 3, 513, 513))
        image = image.cuda()
        output = model(image)
        pred_s = output.data.cpu().numpy()
        pred_r = np.argmax(pred_s, axis=1)
        file_a.write('all' + ' ' + str(pred_s) + ' ' + str(pred_r) + '\n')
        for ii in range(len(files)):
            mergeArr_list_tU_bk = copy.deepcopy(mergeArr_list_tU)
            del mergeArr_list_tU_bk[ii]
            tU_arr_quit = np.zeros((512, 512, 3))
            for jj in range(len(mergeArr_list_tU_bk)):
                tU_arr_quit = tU_arr_quit + mergeArr_list_tU_bk[jj]
            tU_arr_quit = tU_arr_quit / len(mergeArr_list_tU_bk)
            tU_quit_pil = Image.fromarray(tU_arr_quit.astype(np.uint8)).convert('RGB')
            sample = {'image': tU_quit_pil}
            sample = transform_te(sample)
            image = sample['image']
            image = image.reshape((1, 3, 513, 513))
            image = image.cuda()
            output = model(image)
            pred_s = output.data.cpu().numpy()
            pred_r = np.argmax(pred_s, axis=1)
            file_a.write(files[ii] + ' ' + str(pred_s) + ' ' + str(pred_r) + '\n')
