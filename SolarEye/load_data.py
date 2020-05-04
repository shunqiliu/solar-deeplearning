import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import nn_Classes

def load_pic():
    """
    Load pictures
    Shape: (45754, 3, 192, 192)--(Batch size, Channel, Height, Width)
    """
    return np.load('image.npy',mmap_mode='r')

def load_label():
    """
    Load power loss
    Shape: (45754,)--(Batch size)
    """
    return np.load('label.npy',mmap_mode='r')

def load_feats():
    """
    Load environmental features
    Shape:(45754,4)--(Batch size,(Hour,Min,Sec,Irradiance level))
    """
    return np.load('feats.npy',mmap_mode='r')

def load_all():
    """
    Load images, power loss and environmental features individually
    Return:
        images: Solar panel images
        Power_loss: %age power_loss of the panel with respect to the clean panel
        feats: environmental features
    """
    images=load_pic().transpose((0,3,1,2))
    Power_loss=load_label()
    feats=load_feats()
    return images,Power_loss,feats

def load_set():
    """
    Load the data as datasets
    Return:
        d_train: train set, includes 27537 data
        d_test: test set, includes 18217 data
    """
    i,l,r=load_all()
    print('Load all')
    indt=np.random.randint(45754,size=18217)
    indr=np.random.randint(45754,size=27537)
    print('begin random')
    d_train=nn_Classes.SolarNet(i[indr],l[indr],r[indr])
    d_test=nn_Classes.SolarNet(i[indt],l[indt],r[indt])
    return d_train,d_test
