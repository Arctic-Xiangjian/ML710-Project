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


sys.path.append("../ML710-Project/pretrain")
import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import misc, lamb
from utils.imagenet import build_imagenet_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups


enc: encoder.SparseEncoder = build_sparse_encoder('resnet50', input_size=224, sbn=True, drop_path_rate=0.0, verbose=False)
dec = LightDecoder(enc.downsample_raito, sbn=True)

model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=0.6,
        densify_norm='bn', sbn=True, hierarchy=4,
    )

BATCH_SIZE = 128
PATH_DATA = '/home/xiangjian/project'
SEED = 42
DEVICE = 3
EPOCHS = 100
WANDB_API = '8aa455e04a4782e07ec03e938370c9a90f364deb'