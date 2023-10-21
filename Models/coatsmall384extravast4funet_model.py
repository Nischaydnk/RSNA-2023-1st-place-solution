import torch
from torch import nn, optim
import torch.nn.functional as F

import timm
import segmentation_models_pytorch as smp

import sys
sys.path.append('./Configs/')
from coatsmall384extravast4funet_cfg import CFG

import sys
sys.path.append('./')
from paths import PATHS

import sys
sys.path.append(f'{PATHS.CONTRAIL_MODEL_BASE}')

from collections import OrderedDict
from src.coat import CoaT,coat_lite_mini,coat_lite_small,coat_lite_medium
from src.layers import *

from transformers import get_cosine_schedule_with_warmup

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
        
        self.lstm2 = nn.GRU(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.15),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, 2),
        )
        
        if pre is not None:
            sd = torch.load(pre)['model']
            print(self.enc.load_state_dict(sd,strict=False))
        
        self.lstm = nn.ModuleList([LSTM_block(nc[-2]),LSTM_block(nc[-1])])
        self.dec4 = UnetBlock(nc[-1],nc[-2],384)
        self.dec3 = UnetBlock(384,nc[-3],192)
        self.dec2 = UnetBlock(192,nc[-4],96)
        self.fpn = FPN([nc[-1],384,192],[32]*3)
        self.drop = nn.Dropout2d(ps)
        self.final_conv = nn.Sequential(UpBlock(96+32*3, 4, blur=True))
        self.up_result=2
    
    def forward(self, x):
        x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        encs = self.enc(x)
        encs = [encs[k] for k in encs]
        dec4 = encs[-1]

        if self.mask_head:
            dec3 = self.dec4(dec4,encs[-2])
            dec2 = self.dec3(dec3,encs[-3])
            dec1 = self.dec2(dec2,encs[-4])
    
            # print(dec4.shape)
            x = self.fpn([dec4, dec3, dec2], dec1)
            x = self.final_conv(self.drop(x))
            if self.up_result != 0: x = F.interpolate(x,scale_factor=self.up_result,mode='bilinear')

        feat = dec4
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        
        feat, _ = self.lstm2(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, x,x
        else:
            return feat, None


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.318]).cuda()).cuda()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        
    def forward(self, outputs, targets, masks_outputs, masks_outputs2, masks_targets):
        loss1 = self.bce(outputs, targets.float())
        
        
        # any_greater_than_zero = torch.any(targets[:,:,:3] > 0, dim=1)
        # targets2 = any_greater_than_zero.int()
        # targets2 = torch.max(any_greater_than_zero.int(), 1).values
        
        masks_outputs = masks_outputs.float()
        masks_outputs2 = masks_outputs2.float()
        
        masks_targets = masks_targets.float().flatten(0, 1)
        
        loss2 = self.dice(masks_outputs, masks_targets) + self.dice(masks_outputs2, masks_targets)
        
        
        loss = loss1 + (loss2 * 0.125) 
        
        return loss

def define_criterion_optimizer_scheduler_scaler(model, CFG):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler