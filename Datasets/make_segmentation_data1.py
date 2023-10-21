import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import copy
import time

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import pydicom
import nibabel as nib

import sys
sys.path.append('./')
from paths import PATHS


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


#UTILS

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
    
    #img = get_standardized_pixel_array(dcm)
    
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
        
        if np.min(image)<0:
            image = image + np.abs(np.min(image))
        
        image = image / image.max()
        
        volume.append(image)
        
    return np.stack(volume)

def load_segmentation_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[:, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume

def load_total_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[::-1, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume

def load_total_segmentation_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[::-1, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume

print(PATHS.BASE_PATH, PATHS.OUTPUT_BASE)

seg_studies = [x.split('.')[0] for x in os.listdir(PATHS.BASE_PATH + '/segmentations/')]

study_level = pd.read_csv(f'{PATHS.BASE_PATH}/train.csv')

patient_to_studies = {}
for pat in study_level.patient_id:
    patient_to_studies[pat] = os.listdir(f"{PATHS.BASE_PATH}/train_images/{pat}/")
    
study_level['seg_study'] = study_level.patient_id.apply(lambda x: max([1 if y in seg_studies else 0 for y in patient_to_studies[x]]))
study_level['seg_studies'] = study_level.patient_id.apply(lambda x: [y for y in patient_to_studies[x]])

seg_study_level = study_level[study_level['seg_study']==1]

print("MAKING SEGMENTOR DATA")

SAVE_FOLDER = PATHS.SEGMENTOR_SAVE_FOLDER
os.makedirs(SAVE_FOLDER, exist_ok=1)

jump = 16
seq = 32
sz = 128

for i, row in tqdm(seg_study_level.iterrows()):
    patient = row.patient_id
    studies = row.seg_studies

    for study in studies:
    
        dcms = glob_sorted(f"{PATHS.BASE_PATH}/train_images/{patient}/{study}/*.dcm")
            
        segmentation_path = f'{PATHS.BASE_PATH}/segmentations/{study}.nii'
        
        volume = load_volume(dcms)
        seg_volume = load_segmentation_volume(segmentation_path)
        
        seg_volume = seg_volume.astype(np.uint8)
            
        volume = volume[np.arange(0, volume.shape[0], 2)]

        volume = np.stack([cv2.resize(x, (sz, sz)) for x in volume])

        seg_volume = seg_volume[np.arange(0, seg_volume.shape[0], 2)].copy()

        seg_volume = np.stack([cv2.resize(x, (sz, sz), interpolation=cv2.INTER_NEAREST_EXACT) for x in seg_volume])
        
        volumes, seg_volumes = [], []
        cuts = [(x, x+seq) for x in np.arange(0, volume.shape[0], jump)[:-2]]
        
        if cuts:
            for cut in cuts:
                volumes.append(volume[cut[0]:cut[1]])
                seg_volumes.append(seg_volume[cut[0]:cut[1]])

            volumes, seg_volumes = np.stack(volumes), np.stack(seg_volumes)
        else:
            volumes, seg_volumes = np.zeros((1, seq, sz, sz), dtype=np.uint8), np.zeros((1, seq, sz, sz), dtype=np.uint8)
            volumes[0, :len(volume)] = volume
            seg_volumes[0, :len(seg_volume)] = seg_volume
        
        for v in range(len(volumes)):
            np.save(f"{SAVE_FOLDER}/{patient}_{study}_vol{v}.npy", volumes[v])
            np.save(f"{SAVE_FOLDER}/{patient}_{study}_vol{v}_mask.npy", seg_volumes[v])
    
    #break







#UTILS

def glob_sorted(path):
    return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))

def load_volume(dcms):
    volume = []
    for dcm_path in dcms:
        dcm = pydicom.read_file(dcm_path)
        image = dcm.pixel_array
        volume.append(image)
        
    return np.stack(volume)

def load_segmentation_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[:, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume

def load_total_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[::-1, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume

def load_total_segmentation_volume(path):
    volume = nib.load(path).get_fdata()
    
    volume = volume[::-1, ::-1, ::-1].transpose(2, 1, 0)
    
    return volume


print("MAKING TOTAL-SEGMENTOR DATA")

meta = pd.read_csv(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/meta.csv', delimiter=';')
meta['abdomen'] = meta.study_type.apply(lambda x: 1 if 'abdomen' in x else 0)
meta = meta[meta['abdomen']==1].reset_index(drop=True)

SAVE_FOLDER = PATHS.TOTAL_SEGMENTOR_SAVE_FOLDER
os.makedirs(SAVE_FOLDER, exist_ok=1)

jump = 16
seq = 32
sz = 128

for study in tqdm(meta.image_id):
    try:
        volume = load_total_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/ct.nii.gz')
        seg_volume_1 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/liver.nii.gz')
        seg_volume_2 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/spleen.nii.gz')
        seg_volume_3 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/kidney_left.nii.gz')
        seg_volume_4 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/kidney_right.nii.gz')
        seg_volume_5_1 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/colon.nii.gz')
        seg_volume_5_2 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/duodenum.nii.gz')
        seg_volume_5_3 = load_total_segmentation_volume(f'{PATHS.TOTAL_SEGMENTOR_FOLDER}/{study}/segmentations/small_bowel.nii.gz')
    except:
        continue
    
    seg_volume_5 = np.clip(seg_volume_5_1+seg_volume_5_2+seg_volume_5_3, 0, 1)
    
    seg_volume = np.zeros_like(seg_volume_1)
    seg_volume[seg_volume_1==1] = 1
    seg_volume[seg_volume_2==1] = 2
    seg_volume[seg_volume_3==1] = 3
    seg_volume[seg_volume_4==1] = 4
    seg_volume[seg_volume_5==1] = 5
    
    volume = volume + abs(volume.min())
    volume = volume / volume.max()

    volume = (volume * 255).astype(np.uint8)

    seg_volume = seg_volume.astype(np.uint8)

    volume = volume[np.arange(0, volume.shape[0], 2)]

    volume = np.stack([cv2.resize(x, (sz, sz)) for x in volume])

    seg_volume = seg_volume[np.arange(0, seg_volume.shape[0], 2)].copy()

    seg_volume = np.stack([cv2.resize(x, (sz, sz), interpolation=cv2.INTER_NEAREST_EXACT) for x in seg_volume])

    volumes, seg_volumes = [], []
    cuts = [(x, x+seq) for x in np.arange(0, volume.shape[0], jump)[:-2]]

    if cuts:
        for cut in cuts:
            volumes.append(volume[cut[0]:cut[1]])
            seg_volumes.append(seg_volume[cut[0]:cut[1]])

        volumes, seg_volumes = np.stack(volumes), np.stack(seg_volumes)
    else:
        volumes, seg_volumes = np.zeros((1, seq, sz, sz), dtype=np.uint8), np.zeros((1, seq, sz, sz), dtype=np.uint8)
        volumes[0, :len(volume)] = volume
        seg_volumes[0, :len(seg_volume)] = seg_volume

    for v in range(len(volumes)):
        np.save(f"{SAVE_FOLDER}/{study}_vol{v}.npy", volumes[v])
        np.save(f"{SAVE_FOLDER}/{study}_vol{v}_mask.npy", seg_volumes[v])
    
    #break

