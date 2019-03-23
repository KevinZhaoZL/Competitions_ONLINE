# -*- coding: utf-8 -*-
"""
@author: 赵磊
@project: UGPU_Server
@file: EV_Per_Img_Score_Sort.py
@time: 2019/3/15 9:20
@description:
"""
import os
import pandas as pd

path = "F:/2_Data/ScoresPre/"
csv_path = "F:/2_Data/train_label_1.csv"
outpath = "F:/2_Data/ScoresAfter/"

v_count = 0
dirlist = []  # all dir name
df = pd.read_csv(csv_path)
df_dict = {}
for i in range(df['id'].__len__()):
    df_dict[df['id'][i]] = df['ret'][i]

for _, _, files in os.walk(path):
    for ii, item in enumerate(files):
        print(ii + 1)
        fa = open(outpath + item, 'a+')
        csv_pre = item.replace('.txt', '')
        flag = df_dict[csv_pre]
        file_scores = path + item
        tmp_list = []
        with open(file_scores, 'r') as fs:
            for line in fs:
                tmp = line.strip().split(' [')
                tmp_ = []
                for i in range(len(tmp)):
                    a = tmp[i].replace('[', '')
                    b = a.replace(']', '')
                    tmp_.append(b)
                tmp_list.append(tmp_)

        _name = tmp_list[0][0]
        _score = tmp_list[0][1].strip().split(' ')
        while _score.__len__() != 2:
            del _score[1]
        _score = [float(x) for x in _score]
        _flag = int(tmp_list[0][2])

        if flag == _flag:
            for jj in range(1, len(tmp_list)):
                pic_name = tmp_list[jj][0]
                pic_score = tmp_list[jj][1].strip().split(' ')
                while pic_score.__len__() != 2:
                    del pic_score[1]
                pic_score = [float(x) for x in pic_score]
                pic_flag = tmp_list[jj][2]
                # pic_score-_score
                result = [pic_score[i_] - _score[i_] for i_ in range(0, len(_score))]

                if flag == 0:
                    fa.write(pic_name + ' ' + str(result[0]) + '\n')
                else:
                    fa.write(pic_name + ' ' + str(result[1]) + '\n')
