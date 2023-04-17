import sys
import os
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import timm
from torchgpipe import GPipe


sys.path.append("../ML710-Project")
sys.path.append("../ML710-Project/model_and_hyperpam")

from dataset.classification_dataset import (
    train_datasets,
)

from model_and_hyperpam import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    WANDB_API,
)

from parallel.gpipe_class import ViTSequential

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = BATCH_SIZE //2
EPOCHS = EPOCHS // 2

device_ids = ['cuda:0', 'cuda:1']  # 在这里设置要使用的GPU设备ID
model_name = 'vit_large_patch16_224'
num_classes = 5
model_ = ViTSequential(num_classes=num_classes)



wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='gpipe_2gpu'+'_'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_loader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


group_size = 2
num_groups = torch.cuda.device_count() // group_size
devices = [list(range(i*group_size, (i+1)*group_size)) for i in range(num_groups)]

model = GPipe(
    model_,
    balance=[1] * (num_groups - 1) + [0],
    devices=devices,
    chunks=num_groups,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

model.train()
for epoch in tqdm(range(EPOCHS)):
    this_epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        this_epoch_loss += loss.item()
    this_epoch_loss /= len(train_loader)
    wandb.log({"loss": this_epoch_loss})
        
