import sys
sys.path.append('./')
from paths import PATHS
import torch

class CFG:
    DDP = 1
    DDP_INIT_DONE = 0
    N_GPUS = 3
    FOLD = 0
    
    model_name = 'resnet18d'
    V = '1'
    
    OUTPUT_FOLDER = f"{PATHS.SEGMENTATION_MODEL_SAVE}/{model_name}_v{V}"
    
    seed = 3407
    
    device = torch.device('cuda')
    
    n_folds = 5
    folds = [i for i in range(n_folds)]

    train_batch_size = 8
    valid_batch_size = 8
    acc_steps = 1
    
    lr = 5e-4
    wd = 1e-6
    n_epochs = 20
    n_warmup_steps = 0
    upscale_steps = 1.05
    validate_every = 1
    
    epoch = 0
    global_step = 0

    autocast = True

    workers = 2


