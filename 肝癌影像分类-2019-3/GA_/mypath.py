# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午10:39
# @Author  : Zhao Lei
# @File    : mypath.py
# @Desc    : 数据路径等

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/yuanye/Moun/1_PROJECTS/1-Python_r/win10_client_zl/psacal_format_LC'
            # return 'F://2_Data//Liver_Cancer_Dataset//dataset_train1//psacal_format_LC'
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError