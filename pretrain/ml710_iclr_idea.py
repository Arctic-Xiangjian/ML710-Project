from datetime import datetime
import math
import sys
import time
from functools import partial
from typing import List
import random
import numpy as np
import os
import wandb
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import misc, lamb
from utils.imagenet import build_imagenet_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups

sys.path.append("../ML710-Project")
sys.path.append("../ML710-Project/model_and_hyperpam")

from model_and_hyperpam import (
    SEED,
    BATCH_SIZE,
    PATH_DATA,
    DEVICE,
    EPOCHS,
    WANDB_API,
    model_without_ddp
)


from dataset.pretrain_dataset_test import train_datasets

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)

device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
print('test')
model_without_ddp = model_without_ddp.to(device)

data_loader_train = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)


param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})

opt_clz = {
    'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
    'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95)),
    'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, 0.95), max_grad_norm=5.0),
}['lamb']

optimizer = opt_clz(params=param_groups, lr=2e-4*BATCH_SIZE/256, weight_decay=0.0)

def pre_train_epochs(num_epochs, itrt_train, iters_train, model, optimizer, device):
    model = model.to(device)
    wandb.login(key=WANDB_API)
    run = wandb.init(project='ml710_project', entity='arcticfox', name='pretrain_ourdata'+'_'+'resnet50'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)
    
    for ep in tqdm(range(num_epochs)):
        model.train()
        me = misc.MetricLogger(delimiter='  ')
        me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
        header = f'[PT] Epoch {ep}:'
        optimizer.zero_grad()
        early_clipping = 5. > 0 and not hasattr(optimizer, 'global_grad_norm')
        late_clipping = hasattr(optimizer, 'global_grad_norm')

        if early_clipping:
            params_req_grad = [p for p in model.parameters() if p.requires_grad]
        # reset iterator every epoch

        for it, (inp, _) in enumerate(data_loader_train):
            # adjust lr and wd
            min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, 2e-4*BATCH_SIZE/256, 0.04, 0.2, it + ep * iters_train, 40 * iters_train, 1600 * iters_train)

            # forward and backward
            inp = inp.to(device, non_blocking=True)
            # SparK.forward (Replace with actual forward function call)
            loss = model(inp, active_b1ff=None, vis=False)
            optimizer.zero_grad()
            loss.backward()
            loss = loss.item()

            if not math.isfinite(loss):
                print(f'Loss is {loss}, stopping training!', force=True, flush=True)
                sys.exit(-1)
            
            wandb.log({"loss": loss})

            # optimize
            grad_norm = None
            if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, 5.).item()
            optimizer.step()
            if late_clipping: grad_norm = optimizer.global_grad_norm
            torch.cuda.synchronize()

            # log
            # me.update(last_loss=loss)
            # me.update(max_lr=max_lr)
            # tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
            # tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
            # tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
            # tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
            # tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')

            # if grad_norm is not None:
            #     me.update(orig_norm=grad_norm)
            #     tb_lg.update(orig_norm=grad_norm, head='train_hp')
            # tb_lg.set_step()

        print(f'Finished training epoch {ep}')

    run.finish()

    return {k: meter.global_avg for k, meter in me.meters.items()}, model


stats,model_without_ddp_new = pre_train_epochs(EPOCHS, itrt_train, iters_train, model=model_without_ddp, optimizer=optimizer, device=device)
