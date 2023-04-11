import torch
import torch.nn as nn
import timm

class ViTModelParallel(nn.Module):
    def __init__(self, model_name, num_classes, device_ids):
        super().__init__()

        self.device_ids = device_ids

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        # move the different parts of the model to different devices
        self.model.patch_embed = self.model.patch_embed.to(device_ids[0])
        self.model.pos_drop = self.model.pos_drop.to(device_ids[0])

        self.model.blocks = nn.ModuleList(
            [block.to(device_ids[i % len(device_ids)]) for i, block in enumerate(self.model.blocks)]
        )

        self.model.norm = self.model.norm.to(device_ids[-1])
        self.model.head = self.model.head.to(device_ids[-1])

    def forward(self, x):
        x = x.to(self.device_ids[0])
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for i, block in enumerate(self.model.blocks):
            x = x.to(self.device_ids[i % len(self.device_ids)])
            x = block(x)

        x = x.to(self.device_ids[-1])
        x = self.model.norm(x)
        # classification embedding
        x = x[:, 0]
        x = self.model.head(x)

        return x