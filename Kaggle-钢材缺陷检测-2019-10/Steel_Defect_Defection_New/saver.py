# -*- coding: utf-8 -*-
# @Time    : 19-10-14 下午3:59
# @Author  : Zhao Lei
# @File    : saver.py
# @Desc    :
import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, checkname):
        self.directory = os.path.join('run',  checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        epoch = state['epoch']
        filename = "checkpoint_epoch" + str(epoch) + ".pth.tar"
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_dice = state['best_dice']
            with open(os.path.join(self.experiment_dir, 'best_dice.txt'), 'w') as f:
                f.write(str(best_dice))
            if self.runs:
                previous_dice = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_dice.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            dice = float(f.readline())
                            previous_dice.append(dice)
                    else:
                        continue
                max_dice = max(previous_dice)
                if best_dice > max_dice:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

