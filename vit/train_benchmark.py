import sys
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

model = timm.create_model('vit_huge_patch16_224', pretrained=True, num_classes=5)

wandb.login(key=WANDB_API)
run = wandb.init(project='ml710_project', entity='arcticfox', name='classification'+'_'+'vit'+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'), job_type="training",reinit=True)


