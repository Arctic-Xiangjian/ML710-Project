import torch
import torch.nn as nn
import timm


class Cut(nn.Module):
    def forward(self, x):
        return x[:, 0]

class ViTSequential(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTSequential, self).__init__()

        model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=num_classes)
        # self.patch_embed = model.patch_embed
        # self.pos_drop = model.pos_drop
        # self.blocks = nn.Sequential(*model.blocks)
        # self.norm = model.norm
        # self.head = model.head
        self.all_layers = nn.Sequential(
            model.patch_embed,
            model.pos_drop,
            model.blocks,
            model.norm,
            Cut(),
            model.head
        )

    def forward(self, x):
        x = self.all_layers(x)
        return x
    