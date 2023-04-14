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

from parallel.model_3_parallel_class import (
    ModelParallelModel,
)

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

model_name = 'vit_large_patch16_224'
num_classes = 5
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model = ModelParallelModel(model=model)

wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='model_method_3_parallel_2gpu'+'_'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

def train_with_pipeline_parallel(model, train_loader, optimizer, criterion, epoch):
    model.train()
    stream_gpu0 = torch.cuda.Stream(device=0)
    stream_gpu1 = torch.cuda.Stream(device=1)

    for ep in tqdm(range(epoch)):
        this_epoch_loss = 0
        last_batch_on_gpu1 = None
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda(0)
            target = target.cuda(1)

            with torch.cuda.stream(stream_gpu0):
                part1_output = model.part1(data)

            # 如果之前的batch已经在GPU 1上，则等待它完成并计算损失
            if last_batch_on_gpu1 is not None:
                with torch.cuda.stream(stream_gpu1):
                    last_output, last_target = last_batch_on_gpu1
                    loss = criterion(last_output, last_target)
                    this_epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            with torch.cuda.stream(stream_gpu1):
                part1_output = part1_output.cuda(1)
                part2_output = model.part2(part1_output)

            last_batch_on_gpu1 = (part2_output, target)

        # 处理最后一个batch
        with torch.cuda.stream(stream_gpu1):
            last_output, last_target = last_batch_on_gpu1
            loss = criterion(last_output, last_target)
            this_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        this_epoch_loss /= len(train_loader)
        wandb.log({"train_loss": this_epoch_loss})
    return model


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

model_new = train_with_pipeline_parallel(model, train_dataloader, optimizer, criterion, EPOCHS)