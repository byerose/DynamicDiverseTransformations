"""

@author: Ying Meng (y(dot)meng201011(at)gmail(dot)com)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow.python.keras.backend import shape
from torchvision import transforms
import torch
from tensorflow.keras.utils import save_img
import glob
from PIL import Image

def channels_last(data):
    """
    Check if the image is in the shape of (?, img_rows, img_cols, nb_channels).
    :param data:
    :return: True if channel info is at the last dimension, False otherwise.
    """
    # the images can be color images or gray-scales.
    assert data is not None

    if len(data.shape) > 4 or len(data.shape) < 3:
        raise ValueError('Incorrect dimensions of data (expected 3 or 4): {}'.format(data.shape))
    else:
        return (data.shape[-1] == 3 or data.shape[-1] == 1)


def channels_first(data):
    """
        Check if the image is in the shape of (?, nb_channels, img_rows, img_cols).
        :param data:
        :return: True if channel info is at the first dimension, False otherwise.
        """
    # the images can be color images or gray-scales.
    assert data is not None

    if len(data.shape) > 4 or len(data.shape) < 3:
        raise ValueError('Incorrect dimensions of data (expected 3 or 4): {}'.format(data.shape))
    elif len(data.shape) > 3:
        # the first dimension is the number of samples
        return (data.shape[1] == 3 or data.shape[1] == 1)
    else:
        # 3 dimensional data
        return (data.shape[0] == 3 or data.shape[0] == 1)


def set_channels_first(data):
    if channels_last(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 3, 1, 2))
        else:
            data = np.transpose(data, (2, 0, 1))

    return data


def set_channels_last(data):
    if channels_first(data):
        if len(data.shape) == 4:
            data = np.transpose(data, (0, 2, 3, 1))
        else:
            data = np.transpose(data, (1, 2, 0))

    return data


def rescale(data, range=(0., 1.)):
    """
    Normalize the data to range [0., 1.].
    :param data:
    :return: the normalized data.
    """
    l_bound, u_bound = range
    # normalize to (0., 1.)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # rescale into specific range
    data = data * (u_bound - l_bound) + l_bound

    return data

def probs2labels(y):
    if len(y.shape) > 1:
        y = [np.argmax(p) for p in y]

    return y

def myTransform(data, trans=None, param=None, datasets='mnist'):
    if(datasets=='mnist'):
        transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif(datasets=='cifar10'):
        transform = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        raise ValueError('{} is not supported.'.format(datasets))
    data = (data-np.min(data))/(np.max(data)-np.min(data))
    if(trans!=None and param!=None):
        data_tmp = trans(data,param)
    else:
        data_tmp = data
    return transform(torch.from_numpy(data_tmp)).numpy()

def saveimg(data,dir,name):# save array to img
    if dir[-1]!='/':
        dir.append('/')
    if (len(data.shape) == 3):
        if (data.shape[0] == 1 or data.shape[0] == 3):
            data = set_channels_first(data)
        save_img(dir+'/'+name+'.png',data)
    elif (len(data.shape) == 4):
        if (data[0].shape[0] == 1 or data[0].shape[0] == 3):
            data = set_channels_first(data)
        for (i,img) in zip(range(data.shape[0]),data):
            save_img(dir + name + str(i)+'.png', img)

def loadimg(dir):
    # 批量读取图片为array
    # 同一文件下的图片维度应当一致
    if dir[-1]!='/':
        dir.append('/')
    filelist = glob.glob(dir+'*')
    array_list= [np.array(Image.open(fname)) for fname in filelist]
    judge = lambda arr : arr[:,:,np.newaxis] if len(arr.shape)<3 else arr
    return np.array([judge(arr) for arr in array_list])
    