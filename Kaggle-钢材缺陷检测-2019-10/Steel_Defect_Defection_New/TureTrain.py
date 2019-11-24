# -*- coding: utf-8 -*-
# @Time    : 19-10-15 下午4:14
# @Author  : Zhao Lei
# @File    : TureTrain.py
# @Desc    :

# -*- coding: utf-8 -*-
# @Time    : 19-10-14 下午3:57
# @Author  : Zhao Lei
# @File    : Train.py
# @Desc    :

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import os
from model_without_ASPP_resnet101 import DeepLab_Res_JPU
import torch.nn as nn
import math
import torch.nn.functional as F
from saver import Saver
from summaries import TensorboardSummary
from metrics import Evaluator
from custome_transform import transform_tr, transform_val
from tqdm import tqdm

##################################read data############################################
colors_RGB = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 0, 255]
])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(colors_RGB):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.uint8)
    return label_mask


class SteelDataset_trainval(Dataset):
    def __init__(self, rootdir, phase='train'):
        self.rootdir = rootdir
        self.phase = phase
        self.imagedir = os.path.join(self.rootdir, "train_images")
        self.labeldir = os.path.join(self.rootdir, "train_labels")

        self.img_ids = []
        self.images = []
        self.labels = []
        with open(os.path.join(os.path.join(self.rootdir, phase + '.txt')), "r") as f:
            lines = f.read().splitlines()
        for ii, line in enumerate(lines):
            _image = os.path.join(self.imagedir, line + ".jpg")
            _cat = os.path.join(self.labeldir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.img_ids.append(line)
            self.images.append(_image)
            self.labels.append(_cat)

        print('Number of images in {}: {:d}'.format(phase, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.phase == "train":
            return transform_tr(sample)
        elif self.phase == 'val':
            return transform_val(sample)
        else:
            return NotImplementedError

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.labels[index])
        _arr_target = np.asarray(_target)
        _pascal_target = Image.fromarray(encode_segmap(_arr_target))
        return _img, _pascal_target


def make_data_loader(rootdir, num_workers, batchsize):
    train_set = SteelDataset_trainval(rootdir, phase='train')
    val_set = SteelDataset_trainval(rootdir, phase='val')
    train_loader = DataLoader(train_set, batch_size=batchsize['train'], num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchsize['val'], num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


###################################learning rate strategy######################################
class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=90, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_dice):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_dice))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


###########################loss function:cross entrophy,focal loss###########################
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            print("ce loss")
            return self.CrossEntropyLoss
        elif mode == 'focal':
            print("focal loss")
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


###########################################data dir############################################
train_df_path = '/home/yuanye/Moun/1_PROJECTS/1-Python_r/Steel_Defect_Defection_New/data/train.csv'
data_folder = "/home/yuanye/Moun/1_PROJECTS/1-Python_r/Steel_Defect_Defection_New/data/"


############################################Trainer###########################################
class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self):
        self.num_workers = 10
        self.batch_size = {"train": 5, "val": 5}
        self.accumulation_steps = 30 // self.batch_size['train']
        self.lr = 0.001
        self.num_epochs = 120
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.model = DeepLab_Res_JPU(output_stride=16, num_classes=5)
        print(self.model)

        self.train_loader, self.val_loader = make_data_loader(data_folder, num_workers=self.num_workers,
                                                              batchsize=self.batch_size)
        self.loss_type = "ce"
        self.criterion = SegmentationLosses(cuda=True).build_loss(mode=self.loss_type)
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.lr * 10}]
        self.optimizer = torch.optim.SGD(train_params, momentum=0.99,
                                         weight_decay=5e-4, nesterov=False)

        self.scheduler = LR_Scheduler('poly', self.lr,
                                      self.num_epochs, len(self.train_loader))
        self.model = self.model.to(self.device)

        # Define Saver
        self.saver = Saver(checkname="DeepLabv3plusRes50")
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # record
        self.train_loss_epoch = open('logs' + '/train_epoch_loss.txt', 'a+')
        self.val_loss_epoch = open('logs' + '/val_epoch_loss.txt', 'a+')
        self.val_miou_epoch = open('logs' + '/val_miou.txt', 'a+')
        self.val_dice_epoch = open('logs' + '/val_dice.txt', 'a+')
        self.val_PA_epoch = open('logs' + '/val_PA.txt', 'a+')
        self.best_dice = 0.0
        # metrics
        self.evaluator = Evaluator(5)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_dice)
            self.optimizer.zero_grad()
            output = self.model(image)
            if self.loss_type == 'lovasz':
                output = F.softmax(output, dim=1)
                loss = self.criterion(output, target)
            elif self.loss_type == 'ce':
                loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.train_loss_epoch.write(str(train_loss))
        self.train_loss_epoch.write('\n')

        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.batch_size["train"] + image.data.shape[0]))
        print('Train Loss: %.3f' % train_loss)

        if True:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_dice': self.best_dice,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            if self.loss_type == 'lovasz':
                out = F.softmax(output, dim=1)
                loss = self.criterion(out, target)
            else:
                loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        mIoU = self.evaluator.Mean_Intersection_over_Union()
        PA=self.evaluator.Pixel_Accuracy()
        #####################以mIoU为指标训练#############################33
        dice = mIoU
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.val_loss_epoch.write(str(test_loss))
        self.val_loss_epoch.write('\n')
        self.writer.add_scalar('val/PA', PA, epoch)
        self.val_loss_epoch.write(str(PA))
        self.val_loss_epoch.write('\n')
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.val_miou_epoch.write(str(mIoU))
        self.val_miou_epoch.write('\n')
        self.val_dice_epoch.write(str(dice))
        self.val_dice_epoch.write('\n')

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.batch_size["val"] + image.data.shape[0]))
        print("mIoU:{}, PA: {}".format(mIoU,PA))
        print('Loss: %.3f' % test_loss)

        new_pred = dice
        if new_pred > self.best_dice:
            is_best = True
            self.best_dice = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_dice': self.best_dice,
            }, is_best)


if __name__ == '__main__':
    trainer = Trainer()
    for epoch in range(0, trainer.num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
    trainer.writer.close()
