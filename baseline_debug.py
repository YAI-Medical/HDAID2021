import os
import sys
import platform
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, RandomSampler, DataLoader
from torchvision import transforms
import torchinfo

from utils.dataset import ImageList

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation import deeplabv3_resnet101

from models.unet import UNet, InceptionUNet
from models.refinenet import refinenet50, refinenet101, refinenet152, rf_lw50, rf_lw101, rf_lw152

from models.loss import BCEDiceIoUWithLogitsLoss2d, BCEDiceIoULoss2d
from utils.lr_scheduler import CosineAnnealingWarmUpRestarts

import uuid
from utils.training import train_one_epoch

device = torch.device('cuda')
root: str = "../echocardiography/"

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256))])

train_a2c = os.path.join(root, 'train', 'A2C')
train_a2c = ImageList.from_path(train_a2c, transform=transform, target_transform=transform)

train_a4c = os.path.join(root, 'train', 'A4C')
train_a4c = ImageList.from_path(train_a4c, transform=transform, target_transform=transform)

val_a2c = os.path.join(root, 'validation', 'A2C')
val_a2c = ImageList.from_path(val_a2c, transform=transform, target_transform=transform)

val_a4c = os.path.join(root, 'validation', 'A4C')
val_a4c = ImageList.from_path(val_a4c, transform=transform, target_transform=transform)

train_datasets = ConcatDataset([train_a2c, train_a4c])
val_datasets = ConcatDataset([val_a2c, val_a4c])

# # Baseline: DeeplabV3 + ResNet101

# # Pretrained Model
net = deeplabv3_resnet101(pretrained=True, progress=False)
net.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
net.classifier = DeepLabHead(2048, 1)
# net.aux_classifier = nn.Sequential()
net.aux_classifier = FCNHead(1024, 1)

# # Non-pretrained Model
# net = deeplabv3_resnet101(pretrained=False, num_classes=6)

trainable_backbone_layers = ['layer4']
for n, p in net.named_parameters():
    if n.startswith('backbone') and n.split('.')[1] not in trainable_backbone_layers:
        p.requires_grad = False

net.to(device)
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
    net.to(device)

print(torchinfo.summary(net, (1, 1, 256, 256)))

# Lazy-eval iterable dataset: do not set sampler or shuffle options
num_epoch = 1

batch_size = 35
num_workers = 1

loss_function = BCEDiceIoUWithLogitsLoss2d()
optimizer_class = torch.optim.Adam
optimizer_config = {'lr': 1e-6}
scheduler_class = CosineAnnealingWarmUpRestarts
scheduler_config = {'T_0': 10, 'T_mult': 2, 'eta_max': 1e-3, 'T_up': 3, 'gamma': 0.5}

train_loader = DataLoader(train_datasets, batch_size, num_workers=num_workers, drop_last=False)
val_loader = DataLoader(val_datasets, batch_size, num_workers=num_workers, drop_last=False)

optimizer = optimizer_class(net.parameters(), **optimizer_config)
lr_scheduler = scheduler_class(optimizer, **scheduler_config)


def load_state_dict(d):
    net.load_state_dict(d['model'])
    optimizer.load_state_dict(d['optimizer'])
    lr_scheduler.load_state_dict(d['lr_scheduler'])


def state_dict():
    from collections import OrderedDict
    d = OrderedDict()
    d['model'] = net.state_dict()
    d['optimizer'] = optimizer.state_dict()
    d['lr_scheduler'] = lr_scheduler.state_dict()
    return d

try:
    print(f"Re-using session: {session_name}")
except NameError:
    session_name = str(uuid.uuid4())
    print(f"Generating session: {session_name}")

checkpoint_dir = f'checkpoint/{session_name}'
os.makedirs(checkpoint_dir, exist_ok=True)

for ep in range(num_epoch):
    train_one_epoch(net, loss_function, optimizer, lr_scheduler, train_loader, val_loader, device, ep, warmup_start=False)
    # Take care of computational resource.
    if ep == num_epoch - 1:
        torch.save(state_dict(), os.path.join(checkpoint_dir, '{}.pt').format(ep))

from utils.evaluation import all_together, draw_confusion_matrix

label_names = [
    "Left Ventricle",
    "Background"
]

_, _, _, _, cm = all_together(net, val_loader, device=device, verbose=True)
draw_confusion_matrix(
    cm[:5, :5], label_names, label_names,
    figsize=(10, 8), title="Left Ventricle Division"
)