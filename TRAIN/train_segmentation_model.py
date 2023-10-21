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
    try:
        torch.cuda.set_device(local_rank)
    except:
        print("error at", local_rank)
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

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import cv2
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


import sys
sys.path.append('./')
from paths import PATHS

import sys
sys.path.append('./Configs/')
from segmentation_config import CFG


OUTPUT_FOLDER = CFG.OUTPUT_FOLDER

CFG.cache_dir = CFG.OUTPUT_FOLDER + '/cache/'
os.makedirs(CFG.cache_dir, exist_ok=1)
    
seed_everything(CFG.seed)


extra_files = np.array(glob(f'{PATHS.TOTAL_SEGMENTOR_SAVE_FOLDER}/*_mask.npy'))
files = np.array(glob(f'{PATHS.SEGMENTOR_SAVE_FOLDER}/*_mask.npy'))

groups = [x.split('/')[-1].split('_')[0] for x in files]

files[:5], files.shape

class AbdDataset(Dataset):
    def __init__(self, data, transforms, is_training):
        self.data = data
        self.transforms = transforms
        self.is_training = is_training
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        path = self.data[i]
        
        volume_path = path.replace('_mask.npy', '.npy')
        
        volume = np.load(volume_path)
        
        _mask = np.load(path)
        
        volume_ = []
        mask_ = []
        if self.transforms:
            for image, mask in zip(volume, _mask):
                np.random.seed(CFG.global_step)
                random.seed(CFG.global_step)
                
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
                volume_.append(image)
                mask_.append(mask)
        
        volume = np.stack(volume_)[:, 0]
        _mask = np.stack(mask_)
        
        mask_volume = np.zeros((5, _mask.shape[0], _mask.shape[1], _mask.shape[2]), dtype=np.float32)
        for c in range(1, 6):
            mask_volume[c-1][_mask==c] = 1
        
        volume = (volume / volume.max()).astype(np.float32)
        
        return {'images': volume,
                'masks': mask_volume,
                'ids': path}
    

folds = [*GroupKFold(n_splits=CFG.n_folds).split(files, groups=groups)]

def get_loaders():
    
    train_df = files[folds[CFG.FOLD][0]]
    valid_df = files[folds[CFG.FOLD][1]]
    
    train_df = np.concatenate([train_df]*10)
    
    extra_df = np.concatenate([extra_files]*3)
    
    train_df = np.concatenate([train_df, extra_df])
    
    train_augs = A.Compose([
        ToTensorV2(),
    ])
    
    valid_augs = A.Compose([
        ToTensorV2(),
    ])
    
    train_dataset = AbdDataset(train_df, train_augs, 1)
    valid_dataset = AbdDataset(valid_df, valid_augs, 0)
    
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
    
    return train_loader, valid_loader

train_loader, valid_loader = get_loaders()

for d in valid_loader: break



import sys
sys.path.append('./Models/')
from segmentation_model import Model, convert_3d, define_criterion_optimizer_scheduler_scaler

if type(CFG.model_name)!=str: CFG.model_name = 'resnet18d'
m = convert_3d(Model())
m.eval()
with torch.no_grad():
    outs = m(d['images'][:2])
_ = [print(o.shape) for o in outs]


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    for step, data in enumerate(bar):
        step += 1
        
        images = data['images'].cuda()
        masks = data['masks'].cuda()
        
        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            masks_outputs = model(images)
        
        loss = criterion(masks_outputs, masks)
        
        running_loss += (loss - running_loss) * (1 / step)
        
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()
        
        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
        
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
    model = convert_3d(Model())
    st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    model.load_state_dict(st)
    
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
            masks = data['masks'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                masks_outputs = model(images)
            
            masks_outputs = masks_outputs.float().sigmoid().detach().cpu().numpy()
            masks = masks.float().detach().cpu().numpy()
            
            masks_outputs = np.array(masks_outputs)
            masks = np.array(masks)
            ids = np.array(ids)
            
            if running_dist:
                dist.barrier()
                np.save(f'{CFG.cache_dir}/masks_outputs_{get_rank()}.npy', masks_outputs)
                np.save(f'{CFG.cache_dir}/masks_{get_rank()}.npy', masks)
                np.save(f'{CFG.cache_dir}/ids_{get_rank()}.npy', ids)

                dist.barrier()
                
                if is_main_process():
                    masks_outputs = np.concatenate([np.load(f"{CFG.cache_dir}/masks_outputs_{_}.npy") for _ in range(CFG.N_GPUS)])
                    masks = np.concatenate([np.load(f"{CFG.cache_dir}/masks_{_}.npy") for _ in range(CFG.N_GPUS)])
                    ids = np.concatenate([np.load(f"{CFG.cache_dir}/ids_{_}.npy") for _ in range(CFG.N_GPUS)])
                
                dist.barrier()
            
            MASKS_OUTPUTS.extend(masks_outputs)
            MASKS_TARGETS.extend(masks)
            IDS.extend(ids)
        
        #if step==10: break

    MASKS_OUTPUTS = np.stack(MASKS_OUTPUTS)
    MASKS_TARGETS = np.stack(MASKS_TARGETS)
    IDS = np.stack(IDS)
    
    if running_dist:
        dist.barrier()
        if is_main_process():
            np.save(f'{OUTPUT_FOLDER}/MASKS_OUTPUTS.npy', np.array(MASKS_OUTPUTS))
            np.save(f'{OUTPUT_FOLDER}/MASKS_TARGETS.npy', np.array(MASKS_TARGETS))
            np.save(f'{OUTPUT_FOLDER}/IDS.npy', np.array(IDS))
            
        dist.barrier()
    
    losses = [smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=False)(torch.as_tensor(MASKS_OUTPUTS[i:i+1]), torch.as_tensor(MASKS_TARGETS[i:i+1])).item() for i in range(len(MASKS_TARGETS))]
    dice = 1-np.mean(losses)
    
    print(f"EPOCH {CFG.epoch+1} | DICE {dice} |")
    
    if debug:
        return dice, np.array(MASKS_OUTPUTS), np.array(MASKS_TARGETS), np.array(IDS)
        
    else:
        return dice

def run(model, get_loaders):
    if is_main_process():
        epochs = []
        scores = []
    
    best_score = float('-inf')
    for epoch in range(CFG.n_epochs):
        CFG.epoch = epoch
        
        train_loader, valid_loader = get_loaders()
        
        train_one_epoch(model, train_loader)
        
        dist.barrier()
        
        if (CFG.epoch+1)%CFG.validate_every==0 or epoch==0:
            score = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", valid_loader)
        
        if is_main_process():
            epochs.append(epoch)
            scores.append(score)
            
            if score > best_score:
                torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                best_score = score
            
            try:
                command.run(['rm', '-r', CFG.cache_dir])
                pass
            except:
                pass
            
            os.makedirs(CFG.cache_dir, exist_ok=1)


CFG.DDP = 1

if __name__ == '__main__' and CFG.DDP:
    
    init_distributed()
    CFG.DDP_INIT_DONE = 1
    
    #important to setup before defining scheduler to establish the correct number of steps per epoch
    train_loader, valid_loader = get_loaders()
    
    model = convert_3d(Model()).cuda()
    
    if is_main_process(): torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
    
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
else:
    train_loader, valid_loader = get_loaders()
    
    model = convert_3d(Model()).cuda()
    
    criterion, optimizer, scheduler, scaler = define_criterion_optimizer_scheduler_scaler(model)
    
    run(model, get_loaders)
    
import sys
sys.exit(0)