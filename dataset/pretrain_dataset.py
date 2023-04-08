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