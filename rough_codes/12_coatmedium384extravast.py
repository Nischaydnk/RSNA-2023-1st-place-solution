#!/usr/bin/env python
# coding: utf-8

# In[7]:


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
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

import torch.distributed as dist


# In[25]:


import sys
sys.path.append('/home/ubuntu/rsna-2023/')
sys.path.append('/home/ubuntu/rsna-2023/src')


# In[8]:


# !pwd


# In[46]:


# !nvidia-smi


# In[10]:


class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 4
    FOLD = 2
    FULLDATA = 0
    
    model_name = 'coatmed384extra'
    V = '1'
    
    OUTPUT_FOLDER = f"/home/ubuntu/rsna-2023/outputs/{model_name}_v{V}"
    
    seed = 3407
    
    device = torch.device('cuda')
    
    n_folds = 4
    folds = [i for i in range(n_folds)]
    
    image_size = [384, 384]
    # image_size = [128, 128]
    
    TAKE_FIRST = 96
    
    NC = 3
    
    train_batch_size = 1
    valid_batch_size = 2
    acc_steps = 2
    
    lr = 1e-4
    wd = 1e-6
    n_epochs = 9
    n_warmup_steps = 0
    upscale_steps = 1.05
    validate_every = 1
    
    epoch = 0
    global_step = 0
    literal_step = 0

    autocast = True

    workers = 6

if CFG.FULLDATA:
    CFG.seed = CFG.FOLD
    
OUTPUT_FOLDER = CFG.OUTPUT_FOLDER
        
CFG.cache_dir = CFG.OUTPUT_FOLDER + '/cache/'
os.makedirs(CFG.cache_dir, exist_ok=1)
    
seed_everything(CFG.seed)


# In[ ]:





# In[22]:


study_level = pd.read_csv('/home/ubuntu/rsna-2023/train.csv')

patient_to_injury = {pat: study_level[study_level.patient_id==pat].any_injury.values[0] for pat in study_level.patient_id.unique()}

cols = ['kidney_healthy', 'liver_healthy', 'spleen_healthy']

for col in cols:
    exec(f'patient_to_{col}' + " = {pat: study_level[study_level.patient_id==pat]."+f"{col}"+".values[0] for pat in study_level.patient_id.unique()}")
    #break

study_level


# In[ ]:





# In[13]:


data = pd.read_csv('/home/ubuntu/rsna-2023//info_data1_processed_v2.csv')

data


# In[ ]:





# In[ ]:





# In[16]:


def glob_sorted(path):
    return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))

def get_standardized_pixel_array(dcm):
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.to_numpy_image()
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    return pixel_array

def get_windowed_image(dcm, WL=50, WW=400):
    resI, resS = dcm.RescaleIntercept, dcm.RescaleSlope
    
    img = dcm.to_numpy_image()
    
    img = get_standardized_pixel_array(dcm)
    
    img = resS * img + resI
    
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    
    return X

def load_volume(dcms):
    volume = []
    for dcm_path in dcms:
        #dcm = pydicom.read_file(dcm_path)
        #image = dcm.pixel_array
        
        dcm = dicomsdl.open(dcm_path)
        
        image = get_windowed_image(dcm)
        #image = dcm.to_numpy_image()
        
        if np.min(image)<0:
            image = image + np.abs(np.min(image))
        
        image = image / image.max()
        
        volume.append(image)
        
    return np.stack(volume)

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    if type(mask_rle)==str:
    
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
            
    return img.reshape(shape)


'''

os.makedirs("/mnt/md0/rsna_abd/info_data1/", exist_ok=1)
for rows in tqdm(valid_volumes):
    
    patient = rows.iloc[0].patient
    study = rows.iloc[0].study
    start = rows.iloc[0].instance
    end = rows.iloc[-1].instance
    
    files = np.array([f"/mnt/md0/rsna_abd/train_images/{row.patient}/{row.study}/{row.instance}.dcm" for i, row in rows.iterrows()])
        
    vol = load_volume(files)
    vol = np.stack([vol[::3], vol[1::3], vol[2::3]], -1)
    #vol = vol.astype(np.float16)
    vol = (vol * 255).astype(np.uint8)
    
    np.save(f"/mnt/md0/rsna_abd/info_data1/{patient}_{study}_{start}_{end}.npy", vol)
    #break

'''

def get_volume_data(data, step=96, stride=1, stride_cutoff=200):
    volumes = []
    
    for gri, grd in data.groupby('study'):
        
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

class AbdDataset(Dataset):
    def __init__(self, data, transforms, is_training):
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        
        rows = self.data[i]
        
        patient = rows.iloc[0].patient
        study = rows.iloc[0].study
        
        start = rows.iloc[0].instance
        end = rows.iloc[-1].instance
        
        rows = rows[:CFG.TAKE_FIRST]
        
        NC = CFG.NC
        
        p = NC/2
        if str(p).endswith('.0'): p = int(p)-1
        else: p = int(p)
        
        study_crop = rows.iloc[0].study_crop
        
        liver_sizes = [row.liver_size for i, row in rows.iterrows()][p::NC]
        spleen_sizes = [row.spleen_size for i, row in rows.iterrows()][p::NC]
        left_kidney_sizes = [row.left_kidney_size for i, row in rows.iterrows()][p::NC]
        right_kidney_sizes = [row.right_kidney_size for i, row in rows.iterrows()][p::NC]
        kidney_sizes = [max([r, l]) for r, l in zip(right_kidney_sizes, left_kidney_sizes)]
        bowel_sizes = [row.bowel_size for i, row in rows.iterrows()][p::NC]
        liver_rles = [row.liver_rle for i, row in rows.iterrows()][p::NC]
        spleen_rles = [row.spleen_rle for i, row in rows.iterrows()][p::NC]
        left_kidney_rles = [row.left_kidney_rle for i, row in rows.iterrows()][p::NC]
        right_kidney_rles = [row.right_kidney_rle for i, row in rows.iterrows()][p::NC]
        bowel_rles = [row.bowel_rle for i, row in rows.iterrows()][p::NC]

        liver_seg = np.stack([rle_decode(rle, (128, 128)) for rle in liver_rles])
        spleen_seg = np.stack([rle_decode(rle, (128, 128)) for rle in spleen_rles])
        left_kidney_seg = np.stack([rle_decode(rle, (128, 128)) for rle in left_kidney_rles])
        right_kidney_seg = np.stack([rle_decode(rle, (128, 128)) for rle in right_kidney_rles])
        kidney_seg = left_kidney_seg + right_kidney_seg
        bowel_seg = np.stack([rle_decode(rle, (128, 128)) for rle in bowel_rles])

        segmentation_volume = np.stack([liver_seg, spleen_seg, kidney_seg, bowel_seg], -1).clip(0, 1).astype(np.float32)
        
        #segmentation_volume = np.stack([liver_seg], -1).clip(0, 1).astype(np.float32)
        
        #ys, xs = np.where(segmentation_volume.max(0).max(-1))
        #try:
        #    y1, y2, x1, x2 = np.min(ys) / 128, np.max(ys) / 128, np.min(xs) / 128, np.max(xs) / 128
        
        #    study_crop = f'[{y1}, {y2}, {x1}, {x2}]'
        #except:
        #    pass
        
        extravasation_injury = [row.extravasation for i, row in rows.iterrows()][p::NC]
        bowel_injury = [row.bowel for i, row in rows.iterrows()][p::NC]
        
        vol = np.load(f"/home/ubuntu/data/data_theo/{patient}_{study}_{start}_{end}.npy")
        #vol = np.load(f"/mnt/md0/rsna_abd/data_type2_v1/{patient}_{study}_{start}_{end}.npy")
        
        vol = vol[:CFG.TAKE_FIRST]
        
        #'''
        h, w = vol.shape[1:]
        y1, y2, x1, x2 = eval(study_crop)
        y1, y2, x1, x2 = int(y1*h), int(y2*h), int(x1*w), int(x2*w)
        #y1, y2, x1, x2 = y1-int(y1*0.1), y2+int(y2*0.1), x1-int(x1*0.1), x2+int(x2*0.1)
        #vol = vol[:, y1:y2, x1:x2]
        
        h, w = segmentation_volume.shape[1:-1]
        y1, y2, x1, x2 = eval(study_crop)
        y1, y2, x1, x2 = int(y1*h), int(y2*h), int(x1*w), int(x2*w)
        #segmentation_volume = segmentation_volume[:, y1:y2, x1:x2]
        #'''
        
        #vol = vol[:95]
        
        #vol = vol.astype(np.float32) / 255
        
        #files = np.array([f"/mnt/md0/rsna_abd/train_images/{row.patient}/{row.study}/{row.instance}.dcm" for i, row in rows.iterrows()])
        
        #vol = load_volume(files)
        
        vols = []
        for i in range(len(vol)//NC):
            vols.append(vol[i*NC:(i+1)*NC])
        vol = np.stack(vols, 0).transpose(0, 2, 3, 1)
        
        #vol[:, :, :, 0] = vol[:, :, :, 1]

        #print(vol.shape)
        
        #vol = vol.reshape(32, 3, vol.shape[1], vol.shape[2])
        
        volume_, mask_volume = [], []
        if self.transforms:
            first = True
            for image, mask in zip(vol, segmentation_volume):
                
                #np.random.seed(CFG.literal_step)
                #random.seed(CFG.literal_step)
                
                image = image.astype(np.float32) / 255
                
                if self.is_training:
                    if first:
                        transformed = self.transforms(image=image, mask=mask)
                        replay = transformed['replay']
                        first = False
                    else:
                        transformed = A.ReplayCompose.replay(replay, image=image, mask=mask)
                else:
                    transformed = self.transforms(image=image, mask=mask)
                
                image = transformed['image']
                mask = transformed['mask']
                
                volume_.append(image)
                mask_volume.append(mask)
        
        volume = np.stack(volume_)
        mask_volume = np.stack(mask_volume).transpose(0, 3, 1, 2)
        
        volume = volume.astype(np.float32)
        
        #volume = np.concatenate([volume, mask_volume], 1).astype(np.float32)
        
        #'''
        cols = ['liver_healthy', 'spleen_healthy', 'kidney_healthy']
        labels = 1 - rows.iloc[0][cols].values.astype(np.float32)
        
        injury_labels = []
        for i, col in enumerate(cols):
            low_injury, high_injury = 0, 0
            if labels[i]:
                low_injury = study_level[study_level.patient_id==patient][col.replace('_healthy', '_low')].values[0]
                high_injury = study_level[study_level.patient_id==patient][col.replace('_healthy', '_high')].values[0]
            injury_labels.append([low_injury, high_injury])
        
        injury_labels = np.concatenate(injury_labels)
        
        # labels = np.concatenate([labels, injury_labels], -1)
        
        # labels = np.stack([labels]*volume.shape[0]) * np.stack([liver_sizes, spleen_sizes, kidney_sizes,
        #                                                         liver_sizes, liver_sizes,
        #                                                         spleen_sizes, spleen_sizes,
        #                                                         kidney_sizes, kidney_sizes], -1)
        
        #'''
        
        #labels = np.stack([labels]*volume.shape[0])
        
        #labels = np.concatenate([labels, np.stack([extravasation_injury, bowel_injury], -1)], -1)
        
        #labels = np.concatenate([labels, np.stack([bowel_injury,], -1)], -1)
        labels = np.stack([bowel_injury], -1)
        labels = labels * np.stack([bowel_sizes], -1)
        
        #'''
        
        #labels = np.stack([labels]*volume.shape[0])
        
        labels = np.concatenate([labels, np.stack([extravasation_injury], -1)], -1)
        # labels = np.stack([extravasation_injury, bowel_injury], -1).astype(np.float32)
        
        #if self.is_training:
        #    labels[labels==0.] = 0.01
        #    labels[labels==1.] = 0.99
        #labels[labels>0.] = 1.
        
        #labels = labels.max(0)
        
        return {'images': volume,
                'labels': labels,
                'masks': mask_volume,
                'ids': f"{patient}_{study}"}


# In[ ]:





# In[21]:


folds = [*GroupKFold(n_splits=CFG.n_folds).split(data, data.any_injury, groups=data.patient)]

def get_loaders():
    
    if not CFG.FULLDATA:
        valid_df = data.iloc[folds[CFG.FOLD][1]]
    else:
        valid_df = data.iloc[folds[0][1]]
    
    step = 100
    valid_volumes = get_volume_data(valid_df, step, stride=2, stride_cutoff=400)
    
    if not CFG.FULLDATA:
    
        train_df = data.iloc[folds[CFG.FOLD][0]]
        
        train_volumes = get_volume_data(train_df, step, stride=2, stride_cutoff=400)
    
        #train_volumes = [vol for vol in train_volumes if vol.liver_size.max() > 0.3]
        #valid_volumes = [vol for vol in valid_volumes if vol.liver_size.max() > 0.3]
    else:
        train_volumes = get_volume_data(data, step, stride=2, stride_cutoff=400)
    
    injury_present = np.array([vol.any_injury.values[0] for vol in train_volumes])
    non_injured = [train_volumes[idx] for idx in np.where(injury_present==0)[0]]
    injured = [train_volumes[idx] for idx in np.where(injury_present==1)[0]]
    idxs = np.random.choice(np.arange(len(non_injured)), len(injured))
    non_injured = [non_injured[idx] for idx in idxs]
    
    train_volumes = non_injured + injured
    
    #train_df = pd.read_csv('/mnt/md0/birdclef23/specs/train.csv')
    #valid_df = pd.read_csv('/mnt/md0/birdclef23/specs/valid.csv')
    
    train_augs = A.ReplayCompose([
        A.Resize(CFG.image_size[0], CFG.image_size[1]),
        #A.RandomResizedCrop(CFG.image_size[0], CFG.image_size[1], scale=[0.9, 1.1], ratio=[0.9, 1.1]),
        A.Perspective(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, limit=(-25, 25)),
        #A.RandomBrightnessContrast(p=0.5),
        #A.Normalize(),
        ToTensorV2(),
    ])
    
    valid_augs = A.Compose([
        A.Resize(CFG.image_size[0], CFG.image_size[1]),
        #A.Normalize(),
        ToTensorV2(),
    ])
    
    train_dataset = AbdDataset(train_volumes, train_augs, 1)
    valid_dataset = AbdDataset(valid_volumes, valid_augs, 0)
    
    if CFG.DDP and CFG.DDP_INIT_DONE:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset, shuffle=True, drop_last=True)
        train_sampler.set_epoch(CFG.epoch) #needed for shuffling?
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, sampler=train_sampler, num_workers=CFG.workers, pin_memory=False, drop_last=True)
        
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset=valid_dataset, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, sampler=valid_sampler, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=CFG.workers, pin_memory=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    CFG.steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)
    
    return train_loader, valid_loader#, train_data, valid_data

train_loader, valid_loader = get_loaders()

# for d in train_loader: break

# #_, axs = plt.subplots(2, 4, figsize=(24, 12))
# _, axs = plt.subplots(1, 4, figsize=(30, 6))
# axs = axs.flatten()
# for img, ax in zip(range(8), axs):
#     try:
#         ax.imshow(d['images'][img][0].numpy()[:3].transpose(1, 2, 0), cmap='gray')
#     except: pass


# In[ ]:





# In[29]:


# !wget https://vcl.ucsd.edu/coat/pretrained/coat_lite_mini_6b4a8ae5.pth


# In[24]:


# !wget https://vcl.ucsd.edu/coat/pretrained/coat_mini_40667eec.pth


# In[47]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from src.coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from src.layers import *

class Model(nn.Module):
    def __init__(self, pre=None, arch='medium', num_classes=4, ps=0,mask_head=True, **kwargs):
        super().__init__()
        if arch == 'mini': 
            self.enc = coat_lite_mini(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'small': 
            self.enc = coat_lite_small(return_interm_layers=True)
            nc = [64,128,320,512]
        elif arch == 'medium': 
            self.enc = coat_lite_medium(return_interm_layers=True)
            nc = [128,256,320,512]
        else: raise Exception('Unknown model') 

        feats = 512
        drop = 0.0
        self.mask_head = mask_head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.2),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed*2, 2),
        )
        
        if pre is not None:
            sd = torch.load(pre)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)

        self.mask_head_3 = self.get_mask_head(nc[-2])
        self.mask_head_4 = self.get_mask_head(nc[-1])

        
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, 4, blur=True))
        self.up_result=2

    @staticmethod
    def get_mask_head(nb_ft):
        """
        Returns a segmentation head.

        Args:
            nb_ft (int): Number of input features.

        Returns:
            nn.Sequential: Segmentation head.
        """
        return nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
    
    def forward(self, x):
        # x = x[:,:,:5].contiguous()
        # nt = x.shape[2]
        # x = x.permute(0,2,1,3,4).flatten(0,1)
        # x = F.interpolate(x,scale_factor=2,mode='bicubic').clip(0,1)
        x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        # print(encs[3].view(-1,*encs[3].shape[1:]).shape)
        # encs = [encs[0].view(-1,*encs[0].shape[1:])[:,-1], 
        #         encs[1].view(-1,*encs[1].shape[1:])[:,-1], 
        #         self.lstm[-2](encs[2].view(-1,*encs[2].shape[1:]))[:,-1],
        #         self.lstm[-1](encs[3].view(-1,*encs[3].shape[1:]))[:,-1]]
        dec4 = encs[-1]

        # print(encs[-1].shape)
        # print(encs[-2].shape)
        

        if self.mask_head:
            masks1 = self.mask_head_4(encs[-1])
            masks1 = F.interpolate(masks1,size=CFG.image_size,mode='bilinear')
            
            masks2 = self.mask_head_3(encs[-2])
            masks2 = F.interpolate(masks2,size=CFG.image_size,mode='bilinear')

            # print(masks1.shape)
            # print(masks2.shape)
            
            # dec3 = self.dec4(dec4,encs[-2])
            # dec2 = self.dec3(dec3,encs[-3])
            # dec1 = self.dec2(dec2,encs[-4])
    
            # # print(dec4.shape)
            # x = self.fpn([dec4, dec3, dec2], dec1)
            # x = self.final_conv(self.drop(x))
            # if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        
        feat = dec4
        avg_feat = self.avgpool(feat)
        # print(avg_feat.shape)
        
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm2(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks1, masks2
        else:
            return feat, None


# In[48]:


if type(CFG.model_name)!=str: CFG.model_name = 'resnet18d'
    
#CFG.model_name = 'tf_efficientnetv2_s_in21ft1k'

# m = Model()
# m.eval()
# with torch.no_grad():
#    outs = m(d['images'][:2, :4])
# _ = [print(o.shape) for o in outs]


# In[32]:


# model = Model(arch='mini',pre = '/home/ubuntu/rsna-2023/coat_lite_mini_6b4a8ae5.pth', mask_head = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df

Injuries = ['bowel_healthy', 'bowel_injury', 
            'extravasation_healthy', 'extravasation_injury', 
            'kidney_healthy', 'kidney_low', 'kidney_high', 
            'liver_healthy', 'liver_low', 'liver_high', 
            'spleen_healthy', 'spleen_low', 'spleen_high', 
            'any_injury']

def calculate_competition_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    group_to_loss = {}
    
    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        
        loss = sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        
        label_group_losses.append(loss)
        group_to_loss[category] = loss
        
    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    
    #injury_cols = [x for x in Injuries if x not in healthy_cols+['any_injury']]
    #top_n_predictions = valid_study_level_pred[injury_cols].apply(lambda row: row.nlargest(3).prod(), axis=1) * 2
    #any_injury_predictions = np.clip((1 - valid_study_level_pred[healthy_cols]).max(axis=1) + top_n_predictions, 0.01, 0.99)
    
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )
    
    group_to_loss['any_injury'] = any_injury_loss

    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses), group_to_loss


def create_training_solution(y_train):
    sol_train = y_train.copy()
    
    # bowel healthy|injury sample weight = 1|2
    sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)
    
    # extravasation healthy/injury sample weight = 1|6
    sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)
    
    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2, np.where(sol_train['kidney_high'] == 1, 4, 1))
    
    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))
    
    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2, np.where(sol_train['spleen_high'] == 1, 4, 1))
    
    # any healthy|injury sample weight = 1|6
    sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    
    #sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 6)
    
    return sol_train

score = calculate_competition_score

def get_metric_score(keys, NO):
    valid_study_level = pd.concat([study_level[study_level.patient_id==int(key)] for key in keys]).reset_index(drop=True)
    
    valid_study_level_pred = valid_study_level.copy()

    #valid_study_level_pred['extravasation_injury'] = valid_study_level['extravasation_injury'].mean() * 6
    #valid_study_level_pred['extravasation_healthy'] = valid_study_level['extravasation_healthy'].mean()

    #valid_study_level_pred['extravasation_injury'] = valid_study_level['extravasation_injury'].mean() * 6
    #valid_study_level_pred['extravasation_healthy'] = valid_study_level['extravasation_healthy'].mean() * 1
    
    valid_study_level_pred['extravasation_injury'] = (study_level['extravasation_injury'].mean() * 3) + (NO[:, 1]) * 4
    valid_study_level_pred['extravasation_healthy'] = study_level['extravasation_healthy'].mean() * 0.5 + (1-NO[:, 1]) * 0.5

    #valid_study_level_pred['extravasation_injury'] = NO[:, 0]
    #valid_study_level_pred['extravasation_healthy'] = 1 - NO[:, 0]
    
    #valid_study_level_pred['bowel_injury'] = valid_study_level['bowel_injury'].mean() * 2
    #valid_study_level_pred['bowel_healthy'] = valid_study_level['bowel_healthy'].mean()
    
    valid_study_level_pred['bowel_injury'] = NO[:, 0]
    valid_study_level_pred['bowel_healthy'] = 1 - NO[:, 0]
    #valid_study_level_pred['bowel_healthy'] = (1 - NO[:, -1]) * 2

    low_w = 1
    high_w = 2

    valid_study_level_pred['liver_healthy'] = valid_study_level['liver_healthy'].mean()
    valid_study_level_pred['liver_low'] = valid_study_level['liver_low'].mean() * low_w
    valid_study_level_pred['liver_high'] = valid_study_level['liver_high'].mean() * high_w
    
    valid_study_level_pred['spleen_healthy'] = valid_study_level['spleen_healthy'].mean()
    valid_study_level_pred['spleen_low'] = valid_study_level['spleen_low'].mean() * low_w
    valid_study_level_pred['spleen_high'] = valid_study_level['spleen_high'].mean() * high_w
    
    valid_study_level_pred['kidney_healthy'] = valid_study_level['kidney_healthy'].mean()
    valid_study_level_pred['kidney_low'] = valid_study_level['kidney_low'].mean() * low_w
    valid_study_level_pred['kidney_high'] = valid_study_level['kidney_high'].mean() * high_w
    
    valid_study_level_pred['extravasation_injury'] = valid_study_level_pred['extravasation_injury'] + valid_study_level_pred['bowel_injury']

    valid_study_level_pred['auto_injury'] = (1 - valid_study_level_pred[['bowel_healthy', 'extravasation_healthy', 'kidney_healthy', 'liver_healthy', 'spleen_healthy']]).max(1)
    
    true = create_training_solution(valid_study_level.copy())
    sub = create_training_solution(valid_study_level_pred.copy())

    return calculate_competition_score(true.copy(), sub.copy(), 'patient_id')

def mixup_data(x, y, alpha=0.5):
    '''Returns mixed inputs and max of targets'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_max = torch.max(y, y[index])
    return mixed_x, y_max

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.318]).cuda()).cuda()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        
    def forward(self, outputs, targets, masks_outputs, masks_outputs2, masks_targets):
        loss1 = self.bce(outputs, targets)
        
        
        # any_greater_than_zero = torch.any(targets[:,:,:3] > 0, dim=1)
        # targets2 = any_greater_than_zero.int()
        # targets2 = torch.max(any_greater_than_zero.int(), 1).values
        
        masks_outputs = masks_outputs.float()
        masks_outputs2 = masks_outputs2.float()
        
        masks_targets = masks_targets.float().flatten(0, 1)
        
        loss2 = self.dice(masks_outputs, masks_targets) + self.dice(masks_outputs2, masks_targets)
        
        
        loss = loss1 + (loss2 * 0.125) 
        
        return loss

def define_criterion_optimizer_scheduler_scaler(model):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler


# In[ ]:





# In[279]:


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    for step, data in enumerate(bar):
        step += 1
        
        images = data['images'].cuda()
        targets = data['labels'].cuda()
        masks = data['masks'].cuda()
        
        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            logits, mask_outputs, mask_outputs2 = model(images)
        
        loss = criterion(logits, targets, mask_outputs, mask_outputs2, masks)
        
        running_loss += (loss - running_loss) * (1 / step)
        
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()
        
        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
        
        CFG.literal_step += 1
        
        #lr = "{:2e}".format(next(optimizer.param_groups)['lr'])
        lr = "{:2e}".format(optimizer.param_groups[0]['lr'])
        
        if is_main_process():
            bar.set_postfix(loss=running_loss.item(), lr=float(lr), step=CFG.global_step)
        
        #if step==10: break
        
        dist.barrier()
    
    if is_main_process():
        #torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}-{CFG.epoch}.pth")
        torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
        
def valid_one_epoch(path, loader, running_dist=True, debug=False):
    # model = Model(pretrained=False, mask_head=False)
    model = Model(arch='medium', pre=None, mask_head=False, )
    st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    model.load_state_dict(st, strict=False)
    
    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    running_loss = 0.
    
    OUTPUTS = []
    TARGETS = []
    MASKS_OUTPUTS = []
    MASKS_TARGETS = []
    IDS = []
    
    for step, data in enumerate(bar):
        with torch.no_grad():
            images = data['images'].cuda()
            targets = data['labels'].cuda()
            masks = data['masks'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                logits, mask_ouputs = model(images)
            
            #logits = logits[:, :, :9]
            #targets = targets[:, :, :9]
            
            outputs = logits.float().sigmoid().detach().cpu().numpy()
            targets = targets.float().detach().cpu().numpy()
            
            targets[targets > 0.] = 1
            
            ids = np.array(ids)
            
            if running_dist:
                dist.barrier()
            
                np.save(f'{CFG.cache_dir}/preds_{get_rank()}.npy', outputs)
                np.save(f'{CFG.cache_dir}/targets_{get_rank()}.npy', targets)
                np.save(f'{CFG.cache_dir}/ids_{get_rank()}.npy', ids)

                dist.barrier()
                
                if is_main_process():
                    outputs = np.concatenate([np.load(f"{CFG.cache_dir}/preds_{_}.npy") for _ in range(CFG.N_GPUS)])
                    targets = np.concatenate([np.load(f"{CFG.cache_dir}/targets_{_}.npy") for _ in range(CFG.N_GPUS)])
                    ids = np.concatenate([np.load(f"{CFG.cache_dir}/ids_{_}.npy") for _ in range(CFG.N_GPUS)])
                
                dist.barrier()
            
            OUTPUTS.extend(outputs)
            TARGETS.extend(targets)
            IDS.extend(ids)
        
        #running_loss += loss.item()
        
        #if step==10: break
    
    OUTPUTS = np.stack(OUTPUTS)
    TARGETS = np.stack(TARGETS)
    IDS = np.stack(IDS)
    
    SAVE_OUTPUTS = OUTPUTS.copy()
    SAVE_TARGETS = TARGETS.copy()
    SAVE_IDS = IDS.copy()
    
    #OUTPUTS = OUTPUTS[:, :, :9]
    #TARGETS = TARGETS[:, :, :9]
    
    if running_dist:
        dist.barrier()
        if is_main_process():
            np.save(f'{OUTPUT_FOLDER}/OUTPUTS_{CFG.FOLD}_last.npy', OUTPUTS)
            np.save(f'{OUTPUT_FOLDER}/TARGETS_{CFG.FOLD}_last.npy', TARGETS)
            #np.save(f'{OUTPUT_FOLDER}/MASKS_OUTPUTS.npy', np.array(MASKS_OUTPUTS))
            #np.save(f'{OUTPUT_FOLDER}/MASKS_TARGETS.npy', np.array(MASKS_TARGETS))
            np.save(f'{OUTPUT_FOLDER}/IDS_{CFG.FOLD}_last.npy', IDS)
            
        dist.barrier()
    
    
    #1 PATIENT CAN HAVE MULTIPLE VOLUMES, EACH VOLUME IS ONLY 96 SLICES
    
    PATIENT_TO_DAT = {}

    keys = []
    for id in np.unique(IDS):
        k = id.split('_')[0]
        PATIENT_TO_DAT[k] = []
        keys.append(k)
    keys = np.unique(keys)

    for output, target, id in zip(OUTPUTS, TARGETS, IDS):

        #if id.split('_')[0] not in valid_patients:
        #    continue
        try:
            PATIENT_TO_DAT[id.split('_')[0]].append([output, target])
        except:
            passz

    NT = []
    NO = []
    for key in keys:
        dat = np.stack(PATIENT_TO_DAT[key])
        dat = dat.transpose(0, 2, 1, 3)
        dat = np.concatenate(dat)
        dat = dat.transpose(1, 0, 2)

        output, target = dat

        #NO.append(output.max(0))

        out = output.max(0)
        #out[:3] = output.mean(0)[:3]
        #out[4] = output.mean(0)[4]
        #out[6] = output.mean(0)[6]
        #out[8] = output.mean(0)[8]

        NO.append(out)
        #NO.append(output.mean(0))
        #NT.append(target.max(0))
        #NO.append(output.mean(0) + output.max(0))
        NT.append(target.max(0))

    NO, NT = np.array(NO), np.array(NT)
    
    non_zero = list(range(NT.shape[-1]))
    zs = []
    for x in non_zero:
        if NT[:, x].sum():
            zs.append(x)
    zs = np.array(zs)
    
    #auc = roc_auc_score(NT[:, zs], NO[:, zs])
    auc = roc_auc_score(NT, NO)
    
    #print(f"EPOCH {CFG.epoch+1} | ACC {acc} | BACC {bacc} | ACCL {accl} | ACCH {acch}")
    
    metric_score, organ_level_score = get_metric_score(keys, NO)
    
    metric_score = organ_level_score['extravasation']
    
    print(f"EPOCH {CFG.epoch+1} | AUC {auc} | LOSS {metric_score}")
    
    for _ in zs:
        print(_, roc_auc_score(NT[:, _], NO[:, _]))
    
    print(organ_level_score)
    
    if debug:
        return metric_score, OUTPUTS, TARGETS, IDS
    
    return metric_score

def run(model, get_loaders):
    if is_main_process():
        epochs = []
        scores = []
    
    best_score = float('inf')
    for epoch in range(CFG.n_epochs):
        CFG.epoch = epoch
        
        train_loader, valid_loader = get_loaders()
        
        train_one_epoch(model, train_loader)
        
        dist.barrier()
        
        if (CFG.epoch+1)%CFG.validate_every==0 or epoch==0:
            score, OUTPUTS, TARGETS, IDS = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", valid_loader, debug=True)
        
        if is_main_process():
            epochs.append(epoch)
            scores.append(score)
            
            if score < best_score:
                print("SAVING BEST!")
                torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                best_score = score
                
                np.save(f'{OUTPUT_FOLDER}/OUTPUTS_{CFG.FOLD}_best.npy', OUTPUTS)
                np.save(f'{OUTPUT_FOLDER}/TARGETS_{CFG.FOLD}_best.npy', TARGETS)
                #np.save(f'{OUTPUT_FOLDER}/MASKS_OUTPUTS.npy', np.array(MASKS_OUTPUTS))
                #np.save(f'{OUTPUT_FOLDER}/MASKS_TARGETS.npy', np.array(MASKS_TARGETS))
                np.save(f'{OUTPUT_FOLDER}/IDS_{CFG.FOLD}_best.npy', IDS)
            
            try:
                command.run(['rm', '-r', CFG.cache_dir])
                pass
            except:
                pass
            
            os.makedirs(CFG.cache_dir, exist_ok=1)


# In[ ]:





# In[ ]:


CFG.DDP = 1

if __name__ == '__main__' and CFG.DDP:
    
    init_distributed()
    CFG.DDP_INIT_DONE = 1
    
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    # model = Model().cuda()
    model = Model(arch='medium',pre = '/home/ubuntu/rsna-2023/models/coat_lite_medium_384x384_f9129688.pth', mask_head = True).cuda()
    '''
    st = torch.load(f"/mnt/md0/rsna_abd/AAA_CLS/TRY11_CLS/tf_efficientnetv2_s_in21ft1k_v1_/{CFG.FOLD}.pth", map_location='cpu')
    
    #keys = list(st.keys())
    #for key in keys:
    #    if 'head' in key:
    #        st.pop(key)
    
    model.load_state_dict(st, strict=False)
    #'''
    
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    
    #model.load_state_dict(torch.load(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", map_location='cpu'), strict=False)
    
    if is_main_process():
        torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
else:
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = convert_3d(Model()).cuda()
    
    #model.load_state_dict(torch.load(f"/mnt/md0/rsna_spine/AAA_CLS/TRY12_CLS/b5_v1/best_f{CFG.FOLD}.pth", map_location='cpu'), strict=False)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
import sys
sys.exit(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[307]:


folder = 'TRY11_CLS/tf_efficientnetv2_s_in21ft1k_v1_/'

OUTPUTS, TARGETS, IDS = [], [], []

cond = 'best'

for F in range(0, 4):
    outputs = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/OUTPUTS_{F}_{cond}.npy', )
    targets = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/TARGETS_{F}_{cond}.npy', )
    ids = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/IDS_{F}_{cond}.npy', )
    
    OUTPUTS.extend(outputs)
    TARGETS.extend(targets)
    IDS.extend(ids)

OUTPUTS, TARGETS, IDS = np.array(OUTPUTS), np.array(TARGETS), np.array(IDS)
    
UNIVERSAL_IDS = IDS.copy()
    
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[168]:


folder = 'TRY10_CLS/tf_efficientnetv2_s_in21ft1k_v6/'

OUTPUTS, TARGETS, IDS = [], [], []

for F in range(0, 4):
    outputs = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/OUTPUTS_{F}_best.npy', )
    targets = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/TARGETS_{F}_best.npy', )
    ids = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/IDS_{F}_best.npy', )
    
    OUTPUTS.extend(outputs)
    TARGETS.extend(targets)
    IDS.extend(ids)

OUTPUTS, TARGETS, IDS = np.array(OUTPUTS), np.array(TARGETS), np.array(IDS)
    
#UNIVERSAL_IDS = IDS.copy()
    
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[152]:


folder = 'TRY5_CLS/tf_efficientnetv2_s_in21ft1k_v10_newdata/'

OUTPUTS, TARGETS, IDS = [], [], []

for F in range(0, 4):
    outputs = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/OUTPUTS_{F}_best.npy', )
    targets = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/TARGETS_{F}_best.npy', )
    ids = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/IDS_{F}_best.npy', )
    
    OUTPUTS.extend(outputs)
    TARGETS.extend(targets)
    IDS.extend(ids)

OUTPUTS, TARGETS, IDS = np.array(OUTPUTS), np.array(TARGETS), np.array(IDS)
    
#UNIVERSAL_IDS = IDS.copy()
    
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[161]:


folder = 'TRY5_CLS/tf_efficientnetv2_s_in21ft1k_v10/'

OUTPUTS, TARGETS, IDS = [], [], []

for F in range(0, 3):
    outputs = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/OUTPUTS_{F}_best.npy', )
    targets = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/TARGETS_{F}_best.npy', )
    ids = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/IDS_{F}_best.npy', )
    
    OUTPUTS.extend(outputs)
    TARGETS.extend(targets)
    IDS.extend(ids)

OUTPUTS, TARGETS, IDS = np.array(OUTPUTS), np.array(TARGETS), np.array(IDS)
    
#UNIVERSAL_IDS = IDS.copy()
    
OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[164]:


folder = 'nischay_models/rsna-extravastmodel-exp1-uncrop'
#folder = 'nischay_models/coatlitemedium-384-exp1'
try:
    OUTPUTS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_OUTPUTS.npy')
    TARGETS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_TARGETS.npy')
    IDS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_IDS.npy')
except:
    OUTPUTS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_OUTPUTS_last.npy')
    TARGETS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_TARGETS_last.npy')
    IDS = np.load(f'/mnt/md0/rsna_abd/AAA_CLS/{folder}/oof_IDS_last.npy')
    
#UNIVERSAL_IDS = IDS.copy()

'''
st, ed = 0.0, 1.

OUTPUTS = OUTPUTS[int(st*len(OUTPUTS)):int(ed*len(OUTPUTS))]
TARGETS = TARGETS[int(st*len(TARGETS)):int(ed*len(TARGETS))]
IDS = IDS[int(st*len(IDS)):int(ed*len(IDS))]
#'''

'''
st, ed = len(OUTPUTS) - 3237, len(OUTPUTS)

OUTPUTS = OUTPUTS[int(st):int(ed)]
TARGETS = TARGETS[int(st):int(ed)]
IDS = IDS[int(st):int(ed)]
#'''

OUTPUTS.shape, TARGETS.shape, IDS.shape


# In[ ]:





# In[308]:


PATIENT_TO_DAT = {}

keys = []
for id in np.unique(UNIVERSAL_IDS):
    k = id.split('_')[0]
    PATIENT_TO_DAT[k] = []
    keys.append(k)
keys = np.unique(keys)
    
for output, target, id in zip(OUTPUTS, TARGETS, IDS):
    
    #if id.split('_')[0] not in valid_patients:
    #    continue
    try:
        PATIENT_TO_DAT[id.split('_')[0]].append([output, target])
    except:
        passz
    
NT = []
NO = []
for key in keys:
    dat = np.stack(PATIENT_TO_DAT[key])
    dat = dat.transpose(0, 2, 1, 3)
    dat = np.concatenate(dat)
    dat = dat.transpose(1, 0, 2)
    
    output, target = dat
    
    #NO.append(output.max(0))
    
    out = output.max(0)
    #out[:3] = output.mean(0)[:3]
    #out[4] = output.mean(0)[4]
    #out[6] = output.mean(0)[6]
    #out[8] = output.mean(0)[8]
    
    NO.append(out)
    #NO.append(output.mean(0))
    #NT.append(target.max(0))
    #NO.append(output.mean(0) + output.max(0))
    NT.append(target.max(0))
    
NO, NT = np.array(NO), np.array(NT)
NO_v1, NT_v1 = NO.copy(), NT.copy()
print(roc_auc_score(NT, NO), roc_auc_score(NT[:, :3], NO[:, :3]))
try:
    print(roc_auc_score(NT[:, 3:], NO[:, 3:]))
except: 
    pass

for _ in range(NT.shape[-1]):
    print(_, roc_auc_score(NT[:, _], NO[:, _]))


# In[ ]:





# In[315]:


valid_study_level = pd.concat([study_level[study_level.patient_id==int(key)] for key in keys]).reset_index(drop=True)
valid_study_level_pred = valid_study_level.copy()

valid_study_level_pred['extravasation_injury'] = valid_study_level['extravasation_injury'].mean() * 6
valid_study_level_pred['extravasation_healthy'] = valid_study_level['extravasation_healthy'].mean()

#valid_study_level_pred['extravasation_injury'] = valid_study_level['extravasation_injury'].mean() * 4 + (NO2[:, 9]) * 4
#valid_study_level_pred['extravasation_healthy'] = valid_study_level['extravasation_healthy'].mean() * 0.5 + (1- NO2[:, 9]) * 0.5

#NO5 = NO3[:len(valid_study_level_pred), -1].copy() #(NO3[:, -1].copy() * 0.5) + (NO4[:, -1].copy() * 0.5)

NO5 = (NO_uncrop[:, 0].copy() + NO_try10_v6[:, 0].copy() ) / 2#(NO_v23[:len(valid_study_level_pred), -1].copy() + NO_v22[:, -1].copy()) / 2
valid_study_level_pred['extravasation_injury'] = (valid_study_level['extravasation_injury'].mean() * 3) + (NO5) * 4
valid_study_level_pred['extravasation_healthy'] = valid_study_level['extravasation_healthy'].mean() * 0.5 + (1-NO5) * 0.5

#valid_study_level_pred['bowel_injury'] = valid_study_level['bowel_injury'].mean() * 2
#valid_study_level_pred['bowel_healthy'] = valid_study_level['bowel_healthy'].mean()

#valid_study_level_pred['extravasation_injury'] = NO5 * 10
#valid_study_level_pred['extravasation_healthy'] = 1 - NO5

#valid_study_level_pred['extravasation_healthy'] = (0.936447410231967*0.5 + (1-NO3[:, 0])*0.5)
#valid_study_level_pred['extravasation_injury'] = (0.3813155386081983*0.5 + NO3[:, 0]*4)


#NO = (NO_v10.copy() + NO_3.copy()) / 2 #(NO_v10 * 0.8) + (NO_stack * 0.2)
#NO = NO_v10_newdata.copy()#
NO = (NO_v1.copy() + NO_v10_newdata.copy()) / 2

valid_study_level_pred['bowel_injury'] = NO[:, -1] * 1
valid_study_level_pred['bowel_healthy'] = 1 - NO[:, -1]
#valid_study_level_pred['bowel_healthy'] = (1 - NO[:, -1]) * 2

low_w = 1
high_w = 2

valid_study_level_pred['liver_healthy'] = 1 - NO[:, 0]
valid_study_level_pred['liver_low'] = NO[:, 3]*low_w #* 0.5
valid_study_level_pred['liver_high'] = NO[:, 4]*high_w #* 0.5

#valid_study_level_pred['kidney_healthy'] = valid_study_level['kidney_healthy'].mean()
#valid_study_level_pred['kidney_low'] = valid_study_level['kidney_low'].mean()
#valid_study_level_pred['kidney_high'] = valid_study_level['kidney_high'].mean()

valid_study_level_pred['spleen_healthy'] = 1 - NO[:, 1]
valid_study_level_pred['spleen_low'] = NO[:, 5]*low_w #* 0.5
valid_study_level_pred['spleen_high'] = NO[:, 6]*high_w #* 0.5

valid_study_level_pred['kidney_healthy'] = (1 - (NO[:, 2]))
valid_study_level_pred['kidney_low'] = (NO[:, 7])*low_w #* 0.5
valid_study_level_pred['kidney_high'] = NO[:, 8]*high_w #* 0.5

valid_study_level_pred['extravasation_injury'] = valid_study_level_pred['extravasation_injury'] + (valid_study_level_pred['bowel_injury'])

valid_study_level_pred['auto_injury'] = (1 - valid_study_level_pred[['bowel_healthy', 'extravasation_healthy', 'kidney_healthy', 'liver_healthy', 'spleen_healthy']]).max(1)

true = create_training_solution(valid_study_level.copy())
sub = create_training_solution(valid_study_level_pred.copy())
sub


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[316]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v1 + v10-newdata ({uncrop + 10-newdata} for extrav)


# In[ ]:





# In[163]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v1 + v10-newdata


# In[ ]:





# In[159]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v1 + v10-newdata


# In[ ]:





# In[151]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v1


# In[ ]:





# In[156]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v10-newdata


# In[ ]:





# In[143]:


calculate_competition_score(true.copy(), sub.copy(), 'patient_id') #v10 + v10


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




