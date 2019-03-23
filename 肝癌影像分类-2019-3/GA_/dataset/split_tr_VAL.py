# -*- coding: utf-8 -*-  
"""
@author: 赵磊
@project: GA_
@file: split_tr_VAL.py
@time: 2019/3/5 13:45
@description: 
"""
from sklearn.model_selection import train_test_split
import pandas as pd


def tr_val_split():
    csv_file = 'F://2_Data//Liver_Cancer_Dataset//train_dataset1_split_tr_val//train_label_1.csv'
    tr_file = 'F://2_Data//Liver_Cancer_Dataset//train_dataset1_split_tr_val//train.txt'
    val_file = 'F://2_Data//Liver_Cancer_Dataset//train_dataset1_split_tr_val//val.txt'
    tr_label_file = 'F://2_Data//Liver_Cancer_Dataset//train_dataset1_split_tr_val//train_label.txt'
    val_label_file = 'F://2_Data//Liver_Cancer_Dataset//train_dataset1_split_tr_val//val_label.txt'
    df = pd.read_csv(csv_file) 
    length = df['id'].__len__()
    x_list = df['id'].tolist()
    y_list = df['ret'].tolist()
    train_x, val_x, train_y, val_y = train_test_split(x_list, y_list, test_size=0.3, random_state=0)
    f_tr = open(tr_file, "a+")
    f_tr_label = open(tr_label_file, "a+")
    f_val = open(val_file, "a+")
    f_val_label = open(val_label_file, "a+")
    for i in range(len(train_x)):
        f_tr.write(str(train_x[i])+'\n')
        f_tr_label.write(str(train_y[i]) + '\n')
    for j in range(len(val_x)):
        f_val.write(str(val_x[j])+'\n')
        f_val_label.write(str(val_y[j]) + '\n')

if __name__ == '__main__':
    tr_val_split()
