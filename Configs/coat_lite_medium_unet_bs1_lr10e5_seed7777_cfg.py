import sys
sys.path.append('./')
from paths import PATHS
import torch

class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 2
    FOLD = 3

    # model_name = 'tf_efficientnetv2_s_in21ft1k'
    model_name = 'coat_lite_medium' #'eca_nfnet_l0'
    OUTPUT_FOLDER = f"{PATHS.MODEL_SAVE}/coat_lite_medium_unet_bs1_lr10e5_seed7777"

    V = '-1'

    seed = 7777

    device = torch.device('cuda')
    
    n_folds = 4
    folds = [i for i in range(n_folds)]
    
    # image_size = [320, 320]
    image_size = [384, 384]
    
    TAKE_FIRST = 96
    
    NC = 3

    train_batch_size = 1
    valid_batch_size = 2
    acc_steps = 2
    
    lr = 10e-5
    wd = 1e-6
    n_epochs = 12
    n_warmup_steps = 0
    upscale_steps = 1.2
    validate_every = 1
    
    epoch = 0
    global_step = 0
    literal_step = 0

    autocast = True

    workers = 6