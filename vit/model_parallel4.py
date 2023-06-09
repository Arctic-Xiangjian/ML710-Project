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
)

from model_and_hyperpam import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    WANDB_API,
)

from parallel.model_parallel_class import ViTModelParallel

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

device_ids = ['cuda:0', 'cuda:1','cuda:2','cuda:3']
model_name = 'vit_large_patch16_224'
num_classes = 5
model_parallel = ViTModelParallel(model_name, num_classes, device_ids)

wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='vit_model_parallel_4gpu'+'_'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
# test_dataloader = DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def train(model, train_loader, optimizer, criterion, epoch,device=device_ids):
    model.train()
    for ep in tqdm(range(epoch)):
        this_epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device[0]), target.to(device[-1])
            optimizer.zero_grad()
            output = model(data)
            target = target.to(output.device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            this_epoch_loss += loss.item()
        this_epoch_loss /= len(train_loader)
        wandb.log({"train_loss": this_epoch_loss})

optimizer = optim.AdamW(model_parallel.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model_new = train(model_parallel, train_dataloader, optimizer, criterion, EPOCHS)