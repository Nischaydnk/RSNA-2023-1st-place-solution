import sys
sys.path.append('./')
from paths import PATHS
import torch

class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 3
    FOLD = 3
    
    MODEL_NAME = "coat_lite_medium_384.in1k"

    OUTPUT_FOLDER = f'{PATHS.MODEL_SAVE}/coatmed384_vfull_3nc_exp2_384_seed'
    
    seed = 0
    
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
    
    lr = 1e-4
    wd = 1e-6
    n_epochs = 10
    n_warmup_steps = 0
    upscale_steps = 1
    validate_every = 1
    
    epoch = 0
    global_step = 0
    literal_step = 0
    segw = 0.25
    autocast = True

    workers = 6