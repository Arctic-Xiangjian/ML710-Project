import datetime
import math
import sys
import time
from functools import partial
from typing import List
import random
import numpy as np
import os

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import arg_util, misc, lamb
from utils.imagenet import build_imagenet_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups

seed = 42
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('/home/chong.tian/hc701/HC701-PROJECT')

import os
import numpy as np
import yaml

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

from hc701fed.dataset.EyePACS_and_APTOS import Eye_APTOS
from hc701fed.dataset.messidor import MESSIDOR

from hc701fed.transform.transforms import compose
from hc701fed.dataset.messidor import MESSIDOR


PATH_DATA = '/home/chong.tian/hc701'

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(224, scale=(0.67, 1.0), interpolation=interpolation),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3778, 0.2800, 0.2310], std=[0.2892, 0.1946, 0.2006]),
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(224, interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3778, 0.2800, 0.2310], std=[0.2892, 0.1946, 0.2006]),
])

Eye_APTOS_data_dir_options = {
    'EyePACS': os.path.join(PATH_DATA, 'preprocessed/eyepacs'),
    'APTOS': os.path.join(PATH_DATA, 'preprocessed/aptos'),
}

MESSIDOR_data_dir_options = {
    'messidor2': os.path.join(PATH_DATA, 'preprocessed/messidor2'),
    'messidor_pairs' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_pairs'),
    'messidor_Etienne' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_Etienne'),
    'messidor_Brest-without_dilation' : os.path.join(PATH_DATA, 'preprocessed/messidor/messidor_Brest-without_dilation')
}

APTOS_train = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], mode='train', transform_=train_transforms)
EyePACS_train = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], mode='train', transform_=train_transforms)
MESSIDOR_2_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='train', transform_=train_transforms)
MESSIDOR_pairs_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], mode='train', transform_=train_transforms)
MESSIDOR_Etienne_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], mode='train', transform_=train_transforms)
MESSIDOR_Brest_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], mode='train', transform_=train_transforms)

APTOS_Val = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], mode='val', transform_=train_transforms)
EyePACS_Val = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], mode='val', transform_=train_transforms)
MESSIDOR_2_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='val', transform_=train_transforms)
MESSIDOR_pairs_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], mode='val', transform_=train_transforms)
MESSIDOR_Etienne_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], mode='val', transform_=train_transforms)
MESSIDOR_Brest_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], mode='val', transform_=train_transforms)

APTOS_test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['APTOS'], mode='test', transform_=train_transforms)
EyePACS_test = Eye_APTOS(data_dir=Eye_APTOS_data_dir_options['EyePACS'], mode='test', transform_=train_transforms)
MESSIDOR_2_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='test', transform_=train_transforms)
MESSIDOR_pairs_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_pairs'], mode='test', transform_=train_transforms)
MESSIDOR_Etienne_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Etienne'], mode='test', transform_=train_transforms)
MESSIDOR_Brest_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor_Brest-without_dilation'], mode='test', transform_=train_transforms)

# use all data as training data
train_datasets = ConcatDataset([APTOS_train, EyePACS_train, MESSIDOR_2_train, MESSIDOR_pairs_train, MESSIDOR_Etienne_train, MESSIDOR_Brest_train, APTOS_Val, EyePACS_Val, MESSIDOR_2_Val, MESSIDOR_pairs_Val, MESSIDOR_Etienne_Val, MESSIDOR_Brest_Val, APTOS_test, EyePACS_test, MESSIDOR_2_test, MESSIDOR_pairs_test, MESSIDOR_Etienne_test, MESSIDOR_Brest_test])

batch_size = 128

data_loader_train = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)

enc: encoder.SparseEncoder = build_sparse_encoder('resnet50', input_size=224, sbn=True, drop_path_rate=0.0, verbose=False)
dec = LightDecoder(enc.downsample_raito, sbn=True)

model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=0.6,
        densify_norm='', sbn=True, hierarchy=4,
    )

model_without_ddp = model_without_ddp.to(device)

param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})

opt_clz = {
    'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
    'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95)),
    'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, 0.95), max_grad_norm=5.0),
}['lamb']

optimizer = opt_clz(params=param_groups, lr=2e-4*batch_size/256, weight_decay=0.0)

def pre_train_epochs(num_epochs, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model, optimizer, device):
    model = model.to(device)
    
    for ep in range(num_epochs):
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
            min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, 2e-4*batch_size/256, 0.04, 0.2, it + ep * iters_train, 40 * iters_train, 1600 * iters_train)

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

            # optimize
            grad_norm = None
            if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, 5.).item()
            optimizer.step()
            if late_clipping: grad_norm = optimizer.global_grad_norm
            torch.cuda.synchronize()

            # log
            me.update(last_loss=loss)
            me.update(max_lr=max_lr)
            tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
            tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
            tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
            tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
            tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')

            if grad_norm is not None:
                me.update(orig_norm=grad_norm)
                tb_lg.update(orig_norm=grad_norm, head='train_hp')
            tb_lg.set_step()

        print(f'Finished training epoch {ep}')


        import datetime
        pre_train_save_dir = '/home/chong.tian/hc701/checkpoint_pre_train/still_pre_train'
        fine_tune_save_dir = '/home/chong.tian/hc701/checkpoint_pre_train/for_fine_tune'


        if ep % 200 == 0:
            torch.save(model_without_ddp.state_dict(with_config=True), os.path.join(pre_train_save_dir, f'pre_train_model_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))
            torch.save(model_without_ddp.sparse_encoder.sp_cnn.state_dict(), os.path.join(fine_tune_save_dir, f'fine_tune_model_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))

    return {k: meter.global_avg for k, meter in me.meters.items()},model


tb_lg = misc.TensorboardLogger('/home/chong.tian/hc701/check_tensorboard_pre_train_log', is_master=dist.is_master(), prefix='pt')


stats,model_without_ddp_new = pre_train_epochs(1600, tb_lg, itrt_train, iters_train, model=model_without_ddp, optimizer=optimizer, device=device)

import datetime
pre_train_save_dir = '/home/chong.tian/hc701/checkpoint_pre_train/still_pre_train'
fine_tune_save_dir = '/home/chong.tian/hc701/checkpoint_pre_train/for_fine_tune'

torch.save(model_without_ddp_new.state_dict(with_config=True), os.path.join(pre_train_save_dir, f'pre_train_model_end_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))
torch.save(model_without_ddp_new.sparse_encoder.sp_cnn.state_dict(), os.path.join(fine_tune_save_dir, f'fine_tune_model_end_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pth'))

