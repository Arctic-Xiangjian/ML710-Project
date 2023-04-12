import torch
import torch.nn as nn
import timm

class ModelParallelModel(nn.Module):
    def __init__(self, model):
        super(ModelParallelModel, self).__init__()
        self.part1 = nn.Sequential(model.patch_embed,
                                   model.blocks[:len(model.blocks)//2]).cuda(0)
        self.part2 = nn.Sequential(
            model.blocks[len(model.blocks)//2:],
            model.norm, 
            model.head).cuda(1)

    def forward(self, x):
        x = self.part1(x)
        x = x.cuda(1)
        x = self.part2(x)
        x = x[:, 0, :]
        return x