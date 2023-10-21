import dicomsdl
def __dataset__to_numpy_image(self, index=0):
    info = self.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    self.copyFrameData(index, outarr)
    return outarr
dicomsdl._dicomsdl.DataSet.to_numpy_image = __dataset__to_numpy_image    

    
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
import copy
import time
import math
import command
import random

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 12, 8

from skimage import img_as_ubyte
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import *
from sklearn.metrics import *

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import timm

from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append('./')
from paths import PATHS

data = pd.read_csv(f'{PATHS.INFO_DATA_SAVE}')#[:4752]
data

def get_volume_data(data, step=96, stride=1, stride_cutoff=200):
    volumes = []
    
    for gri, grd in tqdm(data.groupby('study')):
        
        #theo's fix
        #grd.instance = list(range(len(grd)))
        
        idxs = np.argsort(grd.z_pos)
        grd = grd.iloc[idxs]
        grd.instance = list(range(len(grd)))
        
        if len(grd)>stride_cutoff:
            grd = grd[::stride]
        
        take_last = False
        if not str(len(grd)/step).endswith('.0'):
            take_last = True
        
        started = False
        for i in range(len(grd)//step):
            rows = grd[i*step:(i+1)*step]
            
            if len(rows)!=step:
                rows = pd.DataFrame([rows.iloc[int(x*len(rows))] for x in np.arange(0, 1, 1/step)])
            
            volumes.append(rows)
            
            started = True
        
        if not started:
            rows = grd
            rows = pd.DataFrame([rows.iloc[int(x*len(rows))] for x in np.arange(0, 1, 1/step)])
            volumes.append(rows)
            
        if take_last:
            rows = grd[-step:]
            if len(rows)==step:
                volumes.append(rows)

        #break

    return volumes


OUTPUT_FOLDER = f'{PATHS.THEO_SAVE_PATH}'
os.makedirs(f"{OUTPUT_FOLDER}/", exist_ok=1)

volume_data = get_volume_data(data, step=100, stride=2, stride_cutoff=400)

def process(i):
    
    rows = volume_data[i]
    
    patient = rows.iloc[0].patient
    study = rows.iloc[0].study
    start = rows.iloc[0].instance
    end = rows.iloc[-1].instance
    
    files = np.array([f"{PATHS.THEO_DATA_PATH}/{row.patient}_{row.study}_" + "0"*(4-len(str(row.instance))) + f"{row.instance}.png" for i, row in rows.iterrows()])
    
    vol = np.stack([cv2.imread(file)[:, :, 0] for file in files])
    
    np.save(f"{OUTPUT_FOLDER}/{patient}_{study}_{start}_{end}.npy", vol)
    
    return None

import multiprocessing as mp
start = 0
with mp.Pool(processes=8) as pool:
    idxs = list(range(start, len(volume_data)))
    imap = pool.imap(process, idxs)
    _ = list(tqdm(imap, total=len(volume_data)-start))