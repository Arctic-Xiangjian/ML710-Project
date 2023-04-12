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

def generate_micro_batches(dataloader, micro_batch_size):
    micro_batch = []
    for data, target in dataloader:
        micro_batch.append((data, target))
        if len(micro_batch) == micro_batch_size:
            yield micro_batch
            micro_batch = []
    if micro_batch:
        yield micro_batch


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
micro_batch_size = BATCH_SIZE // 4

model_name = 'vit_large_patch16_224'
num_classes = 5
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model = ModelParallelModel(model=model)

wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='model_method_3_parallel_2gpu'+'_'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)

train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for ep in tqdm(range(EPOCHS)):
    this_epoch_loss = 0
    for micro_batches in generate_micro_batches(train_dataloader, micro_batch_size):
        # 初始化梯度
        optimizer.zero_grad()

        # 计算micro batch的损失和梯度
        for data, target in micro_batches:
            data, target = data.cuda(0), target.cuda(1)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            this_epoch_loss += loss.item()

        # 更新参数
        optimizer.step()

    this_epoch_loss /= len(train_dataloader)
    wandb.log({"train_loss": this_epoch_loss})