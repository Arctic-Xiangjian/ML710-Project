import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

import os
import sys
sys.path.append('../SparK')
from dataset.EyePACS_and_APTOS import Eye_APTOS
from dataset.messidor import MESSIDOR

from model_and_hyperpam import (
    PATH_DATA,
)

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

MESSIDOR_data_dir_options = {
    'messidor2': os.path.join(PATH_DATA, 'preprocessed/messidor2'),
}


MESSIDOR_2_train = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='train', transform_=train_transforms)



MESSIDOR_2_Val = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='val', transform_=train_transforms)


MESSIDOR_2_test = MESSIDOR(data_dir=MESSIDOR_data_dir_options['messidor2'], mode='test', transform_=train_transforms)


# use all data as training data
train_datasets = ConcatDataset([MESSIDOR_2_train, MESSIDOR_2_Val, MESSIDOR_2_test])