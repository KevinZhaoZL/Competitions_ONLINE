# -*- coding: utf-8 -*-
# @Time    : 19-1-10 下午10:40
# @Author  : Zhao Lei
# @File    : __init__.py.py
# @Desc    :

from dataloaders.datasets import pascal
from torch.utils.data import DataLoader


def make_data_loader_test(**kwargs):
    test_set = pascal.VOCSegmentation_Test(split='test')
    num_class = 2
    return DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)


def make_data_loader(args, **kwargs):
    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, num_class

    else:
        raise NotImplementedError
