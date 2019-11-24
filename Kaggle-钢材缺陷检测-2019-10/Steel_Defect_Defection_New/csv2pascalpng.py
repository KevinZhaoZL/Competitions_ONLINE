# -*- coding: utf-8 -*-
# @Time    : 19-10-14 下午3:57
# @Author  : Zhao Lei
# @File    : csv2pascalpng.py
# @Desc    :
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
import random
from torch.utils.data import DataLoader, Dataset
import os
from skimage.io import imsave

##################################Data Augumentation#############################################
from tqdm import tqdm


def transform_tr(sample):
    composed_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomGaussianBlur(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])

    return composed_transforms(sample)


def transform_val(sample):
    composed_transforms = transforms.Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()])

    return composed_transforms(sample)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}


################################################################################################
##############################################################################
colors_RGB = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 0, 255]
])
colors_RGB_tek = np.array([
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 0, 255]
])


def mask2rle(img):
    '''
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks_4c = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    mask_3c = np.zeros(256 * 1600, dtype=np.uint8)
    mask_3c = mask_3c.tolist()
    for ii, __tmp in enumerate(mask_3c):
        mask_3c[ii] = colors_RGB[0]

    for cat, label in enumerate(labels.values):
        if label is not np.nan and cat < 4:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask_4c = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask_4c[pos:(pos + le)] = 1
                ammama = colors_RGB_tek[cat]
                for le_item in range(le):
                    a = pos + le_item
                    if a >= 409600:
                        a = 409600 - 1
                    mask_3c[a] = ammama

            masks_4c[:, :, cat] = mask_4c.reshape(256, 1600, order='F')
    mask_3c = np.array(mask_3c)
    masks_3c = np.reshape(mask_3c, (256, 1600, 3), order='F').astype(np.float32)
    return fname, masks_4c, encode_segmap(masks_3c)


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


def decode_segmap(label_mask, png_path, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 5
    label_colours = colors_RGB

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        imsave(png_path, rgb)
        # plt.imshow(rgb)
        # plt.show()
    else:
        return rgb


################################################################################################
class SteelDataset(Dataset):
    def __init__(self, df, data_folder, phase):
        self.df = df
        self.root = data_folder
        self.phase = phase
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        image_id, mask_4c, mask_3c = make_mask(idx, self.df)

        image_path = os.path.join(self.root, "train_images", image_id)
        png_path = os.path.join(self.root, "pngs", image_id[:-4] + ".png")
        decode_segmap(mask_3c, png_path, plot=True)
        img = Image.open(image_path).convert('RGB')
        mask_3c = Image.fromarray(mask_3c)

        sample = {'image': img, 'label': mask_3c}
        if self.phase == "train":
            augmented = transform_tr(sample)
        else:
            augmented = transform_val(sample)
        return augmented

    def __len__(self):
        return len(self.fnames)


################################################################################################
def provider(
        data_folder,
        df_path,
        phase,
        batch_size=8,
        num_workers=8,
):
    df = pd.read_csv(df_path)
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    df = train_df if phase == "train" else val_df
    image_dataset = SteelDataset(df, data_folder, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


####################################################################################
train_df_path = '/home/yuanye/Moun/1_PROJECTS/1-Python_r/Steel_Defect_Defection_New/data/train.csv'
data_folder = "/home/yuanye/Moun/1_PROJECTS/1-Python_r/Steel_Defect_Defection_New/data/"


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self):
        self.num_workers = 8
        self.batch_size = {"train": 8, "val": 8}
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.train_loader = self.dataloaders["train"]
        self.val_loader = self.dataloaders["val"]

    def labels2pascalpng(self):
        x = self.train_loader
        tbar = tqdm(x, desc='\r')
        for i, sample in enumerate(tbar):
            print()
        y = self.val_loader
        vbar = tqdm(y, desc='\r')
        for i, sample in enumerate(vbar):
            print()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.labels2pascalpng()
