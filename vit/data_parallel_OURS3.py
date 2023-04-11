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


GPU_IDS = ['cuda:0', 'cuda:1', 'cuda:2']
BATCH_SIZE = BATCH_SIZE // 2 * len(GPU_IDS)
EPOCHS = EPOCHS // 2

model_1 = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=5)
model_2 = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=5)
model_3 = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=5)


wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='data_ours_3gpu'+'_'+'vit'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
# test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def train(model_groups, train_dataloader, optimizers, criterion, num_epochs, device_groups):
    for model in model_groups:
        model.train()
    for model, device in zip(model_groups, device_groups):
        model.to(device)
    for epoch in tqdm(range(num_epochs)):
        this_epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data_group1, data_group2 , data_group3 = data.split(data.shape[0] // 3, dim=0)

            data_group1 = data_group1.to(device_groups[0])
            data_group2 = data_group2.to(device_groups[1])
            data_group3 = data_group3.to(device_groups[2])

            output_group1 = model_groups[0](data_group1)
            output_group2 = model_groups[1](data_group2)
            output_group3 = model_groups[2](data_group3)

            # make output_group1 and output_group2 in the same device
            output_group2 = output_group2.to(output_group1.device)
            output_group3 = output_group3.to(output_group1.device)
            output = torch.cat([output_group1, output_group2, output_group3], dim=0)
            target = target.to(output.device)

            loss = criterion(output, target)

            optimizers[0].zero_grad()
            optimizers[1].zero_grad()
            optimizers[2].zero_grad()
            loss.backward()
            optimizers[0].step()
            optimizers[1].step()
            optimizers[2].step()
            this_epoch_loss += loss.item()

        this_epoch_loss /= len(train_dataloader)
        wandb.log({'loss': this_epoch_loss})

            
optimizer_group1 = optim.AdamW(model_1.parameters(), lr=1e-3)
optimizer_group2 = optim.AdamW(model_2.parameters(), lr=1e-3)
optimizer_group3 = optim.AdamW(model_3.parameters(), lr=1e-3)

optimizers = [optimizer_group1, optimizer_group2, optimizer_group3]

criterion = nn.CrossEntropyLoss()

train([model_1, model_2, model_3], train_dataloader, optimizers, criterion, EPOCHS, GPU_IDS)