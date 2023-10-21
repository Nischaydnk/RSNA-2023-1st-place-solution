import torch
from torch import nn, optim
import torch.nn.functional as F

import timm
import segmentation_models_pytorch as smp

import sys
sys.path.append('./Configs/')
from try11_v8_extrav_cfg import CFG

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
    def __init__(self, pretrained=True, mask_head=True, n_classes=2):
        super(Model, self).__init__()
        
        self.mask_head = mask_head
        
        drop = 0.
        
        true_encoder = timm.create_model(CFG.model_name, pretrained=pretrained, in_chans=CFG.NC,
                                         global_pool='', num_classes=0,
                                         drop_rate=drop, drop_path_rate=drop)

        segmentor = smp.Unet(f"tu-{CFG.model_name}", encoder_weights="imagenet", in_channels=CFG.NC, classes=4)
        self.encoder = segmentor.encoder
        # self.decoder = segmentor.decoder
        # self.segmentation_head = segmentor.segmentation_head
        
        st = true_encoder.state_dict()

        self.encoder.model.load_state_dict(st, strict=False)
        
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        
        feats = true_encoder.num_features
        # feats = 256
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)


        self.mask_head_3 = self.get_mask_head(true_encoder.feature_info[-2]['num_chs'])
        self.mask_head_4 = self.get_mask_head(true_encoder.feature_info[-1]['num_chs'])

        print('total channels in seg head: ',true_encoder.feature_info[-2]['num_chs'],true_encoder.feature_info[-1]['num_chs'] )
        
        lstm_embed = feats * 1
        
        self.lstm = nn.GRU(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, n_classes),
        )

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
    
    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(x)
        # print(i.shape for i in features)
        # print(features[-1].shape)
        # print(features[-2].shape)
        
        if self.mask_head:
            masks1 = self.mask_head_4(features[-1])
            masks1 = nn.functional.interpolate(masks1,size=CFG.image_size,mode='bilinear')
            
            masks2 = self.mask_head_3(features[-2])
            masks2 = nn.functional.interpolate(masks2,size=CFG.image_size,mode='bilinear')
            
            # decoded = self.decoder(*features)
        # 
            # masks2 = self.segmentation_head(decoded)
        
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        # print(feat.shape)
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm(feat)
        
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        
        feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks1, masks2
        else:
            return feat, None
        

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
        
        
        loss = loss1 + (loss2 * 0.15) 
        
        return loss

def define_criterion_optimizer_scheduler_scaler(model, CFG):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler