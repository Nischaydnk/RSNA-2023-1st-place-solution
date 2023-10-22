1.
```
python Datasets/make_segmentation_data1.py
```
2.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_segmentation_model.py
```
3.
```
python Datasets/make_info_data.py
```
4.
```
python Datasets/make_theo_data_volumes.py
```
5.
```
python Datasets/make_our_data_volumes.py
```
6.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatmed384fullseed.py --seed 969696
```
7.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_med_newseg_ourdata_4f.py --fold 1
```
8.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatmed384ourdataseed.py --seed 100
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatmed384ourdataseed.py --seed 6969
```
9.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_v2s_try5_v10_fulldata.py --seed 3407
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_v2s_try5_v10_fulldata.py --seed 123
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_v2s_try5_v10_fulldata.py --seed 123123
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_v2s_try5_v10_fulldata.py --seed 123123123
```
10.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7 --lr 9e-5
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7777 --lr 10e-5
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_lite_medium_bs2_lr_seed.py --seed 7777777 --lr 11e-5
```
11.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_lite_medium_bs2_lr125e6.py
```
12.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_lite_medium_unet_bs1_lr10e5_seed7777.py
```
13.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_tf_efficientnetv2_s_in21ft1k_v1_fulldata.py --seed 3407
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_tf_efficientnetv2_s_in21ft1k_v1_fulldata.py --seed 42
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_tf_efficientnetv2_s_in21ft1k_v1_fulldata.py --seed 69
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_tf_efficientnetv2_s_in21ft1k_v1_fulldata.py --seed 17716124
```
14.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatsmall384extravast4funet.py --fold 1
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatsmall384extravast4funet.py --fold 3
```
15.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_fullextracoatsmall384.py --seed 2024
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_fullextracoatsmall384.py --seed 2717
```
16.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_v8_extrav.py --fold 1
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_v8_extrav.py --fold 2
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_try11_v8_extrav.py --fold 3
```
17.
```
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatmedium384extravast.py --fold 0
```
