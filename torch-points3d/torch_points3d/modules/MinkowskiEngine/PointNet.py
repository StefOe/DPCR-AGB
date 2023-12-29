import MinkowskiEngine as ME
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd

from .common import ACTIVATIONS, GLOBAL_POOL


class MinkowskiPointNet(nn.Module):
    def __init__(self, in_channels, out_channels, activation="relu", global_pool="max", embedding_channel=1024, D=3,
                 dropout=0.0, bn_momentum=.1,
                 **kwargs):
        super().__init__()
        self.act_fn = ACTIVATIONS[activation]()

        self.blocks = nn.Sequential(
            ME.MinkowskiLinear(D + in_channels, 64, bias=False),
            ME.MinkowskiBatchNorm(64, momentum=bn_momentum),
            self.act_fn,

            ME.MinkowskiLinear(64, 128, bias=False),
            ME.MinkowskiBatchNorm(128, momentum=bn_momentum),
            self.act_fn,

            ME.MinkowskiLinear(128, embedding_channel, bias=False),
            ME.MinkowskiBatchNorm(embedding_channel, momentum=bn_momentum),
            self.act_fn,
        )
        self.global_pool = GLOBAL_POOL[global_pool]()

        self.mlp = nn.Sequential(
            ME.MinkowskiLinear(embedding_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512, momentum=bn_momentum),
            self.act_fn,

            ME.MinkowskiLinear(512, 256, bias=False),
            ME.MinkowskiBatchNorm(256, momentum=bn_momentum),
            self.act_fn,
        )
        self.dp1 = ME.MinkowskiDropout(dropout)
        self.final = ME.MinkowskiLinear(256, out_channels, bias=True)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.mlp(x)
        x = self.dp1(x)
        return self.final(x)
