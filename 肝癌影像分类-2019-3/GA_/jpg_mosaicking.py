# -*- coding: utf-8 -*-
"""
@author: 赵磊
@project: UGPU_Server
@file: jpg_mosaicking.py
@time: 2019/3/13 12:32
@description:
"""
import os
from PIL import Image

UNIT_SIZE = 1000


def pinjie(images, num, num_w=4, num_h=3):
    target = Image.new('RGB', (UNIT_SIZE * num_w, UNIT_SIZE * num_h))
    dim1 = []
    dim2 = []
    for n in range(num_w):
        dim1.append(n * UNIT_SIZE)
    for m in range(num_h):
        dim2.append((m + 1) * UNIT_SIZE)

    for i in range(len(images)):
        if i < num_w:
            target.paste(images[i], (dim1[i], 0, (i + 1) * UNIT_SIZE, dim2[0]))
        elif i >= num_w and i < num_w * 2:
            target.paste(images[i], (dim1[i - num_w], UNIT_SIZE, (i + 1 - num_w) * UNIT_SIZE, dim2[1]))
        else:
            target.paste(images[i], (dim1[i - 2 * num_w], 2 * UNIT_SIZE, (i + 1 - 2 * num_w) * UNIT_SIZE, dim2[2]))

    quality_value = 100
    target.save(path + dirlist[num] + '.jpg', quality=quality_value)


# path = "C:/Users/laojbdao/Desktop/FinalResult/result4/different_distribution/"
path = "test_9/"
dirlist = []  # all dir name
for root, dirs, files in os.walk(path):
    for dir in dirs:
        dirlist.append(dir)
num = 0
for dir in dirlist:
    images = []  # images in each folder
    for root, dirs, files in os.walk(path + dir):  # traverse each folder
        print(path + dir + '')

        files_adjust = []
        for ii, item in enumerate(files):
            if ii != 2 and ii != 3:
                files_adjust.append(item)
            elif ii == 2:
                item2 = item
            else:
                item3 = item
        files_adjust.append(item2)
        files_adjust.append(item3)

        for file in files_adjust:
            images.append(Image.open(path + dir + '/' + file))
        pinjie(images, num)
        num += 1
        images = []
