# -*- coding: utf-8 -*-  
"""
@author: 赵磊
@project: GA_
@file: dcm2jpg.py
@time: 2019/3/4 21:46
@description: 
"""

import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


class DCM2JPG():
    def __init__(self):
        self.inpath = 'F://2_Data//Liver_Cancer_Dataset//dataset_test_new//dataset_test_new_dcm'
        self.count = 0
        self.labelpath = 'F://2_Data//Liver_Cancer_Dataset//dataset_test_new//submit_example.csv'

    def dcm2jpg(self, filename):
        ds = pydicom.read_file(filename)  # 读取.dcm文件
        img = ds.pixel_array  # 提取图像信息
        print(img.shape)
        plt.imshow(img)
        # plt.show()
        out_path = filename.replace('dcm', 'jpg')
        out_path = out_path.replace('dataset_test_new_dcm', 'dataset_test_new_jpg')
        plt.imsave(out_path, img)  # 2018/11/6更新，之前保存图片的方式出了一点小问题，对不住，对不住。
        self.count = self.count + 1
        print(self.count)

    def act(self):
        df = pd.read_csv(self.labelpath)
        length = df['id'].__len__()
        for i in range(3000, length):
            personpath = self.inpath + "//" + df['id'][i]
            if not os.path.exists(personpath.replace('dataset_test_new_dcm', 'dataset_test_new_jpg')):
                os.makedirs(personpath.replace('dataset_test_new_dcm', 'dataset_test_new_jpg'))
            tmp_list = os.listdir(personpath)
            for j in range(0, len(tmp_list)):
                dcmpath = os.path.join(personpath, tmp_list[j])
                self.dcm2jpg(dcmpath)

    def mergeJPG(self):
        df = pd.read_csv(self.labelpath)
        length = df['id'].__len__()
        for i in range(3000, length):
            print(i)
            personpath = self.inpath + "//" + df['id'][i]
            personjpgpath = personpath.replace('dataset_test_new_dcm', 'dataset_test_new_jpg')
            tmp_list = os.listdir(personjpgpath)
            tmp = np.zeros((512, 512, 3))
            number = 0
            for j in range(0, len(tmp_list)):
                jpgpath = os.path.join(personjpgpath, tmp_list[j])
                image = Image.open(jpgpath)
                image_arr = np.array(image)
                tmp = tmp + image_arr
                number = number + 1
            tmp = tmp / number
            im = Image.fromarray(tmp.astype(np.uint8)).convert('RGB')
            im.save(personjpgpath.replace('dataset_test_new_jpg', 'dataset_test_new_merge') + ".jpg")


if __name__ == '__main__':
    test = DCM2JPG()
    test.mergeJPG()

