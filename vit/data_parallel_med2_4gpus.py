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

sys.path.append("../ML710-Project")
sys.path.append("../ML710-Project/model_and_hyperpam")

from dataset.classification_dataset import (
    train_datasets,
    val_datasets,
    test_datasets,
)

from model_and_hyperpam import (
    SEED,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    WANDB_API,
)

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)


GPU_IDS = [0,1,2,3]
BATCH_SIZE = BATCH_SIZE // 2 * len(GPU_IDS)
EPOCHS = EPOCHS // 2

model_BASE = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=5)
model = nn.parallel.DistributedDataParallel(model_BASE, device_ids=GPU_IDS, output_device=GPU_IDS[0])

wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='data_pytorch2_4gpu'+'_'+'vit'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
# test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def train(model, train_loader, optimizer, criterion, epoch,device='cuda'):
    model = model.to(device)
    model.train()
    for ep in tqdm(range(epoch)):
        this_epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            this_epoch_loss += loss.item()
        this_epoch_loss /= len(train_loader)
        wandb.log({"train_loss": this_epoch_loss})
    return model

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model_new = train(model, train_dataloader, optimizer, criterion, EPOCHS)