import torch
import torch.nn as nn
import timm

class ModelParallelModel(nn.Module):
    def __init__(self, model):
        super(ModelParallelModel, self).__init__()
        self.part1 = model.patch_embed.cuda(0)
        self.part2 = model.blocks.cuda(1)
        self.part3 = nn.Sequential(model.norm, model.head).cuda(2)

    def forward(self, x):
        x = self.part1(x)
        x = x.cuda(1)
        x = self.part2(x)
        x = x.cuda(2)
        x = self.part3(x)
        x = x[:, 0, :]
        return x