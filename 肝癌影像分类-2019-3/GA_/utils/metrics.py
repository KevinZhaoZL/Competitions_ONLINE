# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午11:03
# @Author  : Zhao Lei
# @File    : metrics.py
# @Desc    :
import pdb

import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.pred = []
        self.target = []

    def add_batch(self, pre_list, targ_list):
        for i in range(len(pre_list)):
            self.pred.append(pre_list[i])
            self.target.append(targ_list[i])

    def R_1_0(self):
        a = 0
        b = 0
        c = 0
        d = 0
        list_target = self.target
        list_pre = self.pred
        for i in range(len(list_target)):
            # 1为正类
            if list_target[i] == 1:
                a = a + 1  # TP+FN
                if list_pre[i] == 1:
                    b = b + 1  # TP
            # 0为正类
            else:
                c = c + 1  # TP+FN
                if list_pre[i] == 0:
                    d = d + 1  # TP
        recall1 = b / (a + 0.1)
        recall0 = d / (c + 0.1)
        return recall1, recall0

    def P_1_0(self):
        a = 0
        b = 0
        c = 0
        d = 0
        list_target = self.target
        list_pre = self.pred
        for i in range(len(list_target)):
            if list_target[i] == 1:
                if list_pre[i] == 1:
                    a = a + 1  # 1TP  0TN
                else:
                    c = c + 1  # 1FN  0FP
            else:
                if list_pre[i] == 1:
                    b = b + 1  # 1FP  0FN
                else:
                    d = d + 1  # 1TN  0TP
        p1 = a / (a + b + 0.1)
        p0 = d / (d + c + 0.1)
        return p1, p0

    def F1_Score(self, r1, r0, p1, p0):
        F1_1 = 2 * p1 * r1 / (r1 + p1 + 0.1)
        F1_0 = 2 * p0 * r0 / (p0 + r0 + 0.1)
        Score = (F1_0 + F1_1) / 2
        return F1_1, F1_0, Score

    def reset(self):
        self.pred = []
        self.target = []
