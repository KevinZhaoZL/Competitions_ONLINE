# -*- coding: utf-8 -*-  
"""
@author: 赵磊
@project: UGPU_Server
@file: test.py
@time: 2019/3/7 10:05
@description: 
"""
import pdb
import random
from tqdm import tqdm

from modeling.img_classifier import *
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np

from modeling.sync_batchnorm import patch_replication_callback
from dataloaders import *

outpath = '/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/experiment_resnetv2_100_sorted10_no_plus/submit_7.csv'
outpath_tmp = '/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/experiment_resnetv2_100_sorted10_no_plus/submit_7.txt'
model = Classifier(num_classes=2,
                backbone='resnetv2',
                output_stride=16,
                sync_bn=None,
                freeze_bn=False)
state_dict_toload = torch.load('/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/model_best.pth.tar')
model.load_state_dict(state_dict_toload.get('state_dict'))
print('trained model loaded!')
model = torch.nn.DataParallel(model, device_ids=[0])
model = model.cuda()
test_loader=make_data_loader_test()


ret_list = []
model.eval()
tbar = tqdm(test_loader, desc='\r')
f=open(outpath_tmp,'a+')
for i, sample in enumerate(tbar):
    image = sample['image']
    image= image.cuda()
    with torch.no_grad():
        output = model(image)
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    ret_list.append(pred[0])
    f.write(str(pred[0])+'\n')
    print(i,pred[0])

column_ret = pd.Series(ret_list, name='ret')
save = pd.DataFrame({'ret': ret_list})
save.to_csv(outpath, index=False, sep=',')
