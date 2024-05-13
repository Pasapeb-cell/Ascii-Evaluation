import torch
import torch.nn as nn
from .vit_model import  VisionTransformer
from .vit_model import ImageToPatches
class VITClasification(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.preprocess = ImageToPatches(patch_size=kwargs["patch_size"])
        self.vit = VisionTransformer(**kwargs)
        self.classification = nn.Sequential(nn.LayerNorm(kwargs["embed_dim"]), nn.Linear(kwargs["embed_dim"], kwargs["num_classes"]))

    def forward(self, x):
        # print(x.shape)
        x = self.preprocess(x)
        x = self.vit(x)
        x = self.classification(x)
        return x