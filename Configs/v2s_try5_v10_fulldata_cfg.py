import sys
sys.path.append('./')
from paths import PATHS
import torch


class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 3
    FOLD = 0
    
    model_name = 'tf_efficientnetv2_s_in21ft1k'
    V = '10_fulldata'
    
    OUTPUT_FOLDER = f"{PATHS.MODEL_SAVE}/{model_name}_v{V}"
    
    seed = 3407
    
    device = torch.device('cuda')
    
    n_folds = 4
    folds = [i for i in range(n_folds)]
    
    image_size = [384, 384]
    
    TAKE_FIRST = 96
    
    NC = 3
    
    train_batch_size = 2
    valid_batch_size = 2
    acc_steps = 2
    
    lr = 2e-4
    wd = 1e-6
    n_epochs = 10
    n_warmup_steps = 0
    upscale_steps = 1.05
    validate_every = 10
    
    epoch = 0
    global_step = 0
    literal_step = 0

    autocast = True

    workers = 4