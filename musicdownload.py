import sys,os,errno,signal,copy
from contextlib import contextmanager

import numpy as np
import musicnet

import torch

root = './musicnet'
checkpoint_path = './checkpoints'
checkpoint = 'musicnet_demo.pt'

try:
    os.makedirs(checkpoint_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
#starting download
train_set = musicnet.MusicNet(root=root, train=True, download=True, window=window, mmap=False)#, pitch_shift=5, jitter=.1)
