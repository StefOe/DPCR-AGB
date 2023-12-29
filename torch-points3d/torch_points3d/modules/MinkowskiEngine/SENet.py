from functools import partial

import MinkowskiEngine as ME
import torch.nn as nn
from MinkowskiEngine import MinkowskiNormalization as N

from .common import ConvNormActivation, MinkowskiLayerNorm, ACTIVATIONS, GLOBAL_POOL
from .resnet_block import BasicBlock, Bottleneck
from .senet_block import SEBasicBlock, SEBottleneck




class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, activation="relu", D=3, first_stride=2, dropout=0.0, drop_path=0.0,
                 bn_momentum=0.1, norm_type="bn", global_pool="mean", use_gn=False, bias=True, **kwargs):
        nn.Module.__init__(self)
        self.D = D
        self.bias = bias
        self.bn_momentum = bn_momentum
        self.cross_dims = []
        self.drop_path = drop_path
        assert self.BLOCK is not None, "BLOCK is not defined"
        assert self.PLANES is not None, "PLANES is not defined"
        assert self.STRIDES is not None, "STRIDES is not defined"

        self.act_fn = ACTIVATIONS[activation]()
        self.norm_type = norm_type
        if norm_type == "bn":
            self.norm_layer = partial(N.MinkowskiBatchNorm, momentum=bn_momentum)
        elif norm_type == "bn_no_affine":
            self.norm_layer = partial(N.MinkowskiBatchNorm, momentum=bn_momentum, affine=False)
        elif norm_type == "in":
            self.norm_layer = N.MinkowskiInstanceNorm
        elif norm_type == "ln":
            self.norm_layer = MinkowskiLayerNorm
        else:
            raise NotImplementedError(f"Choose either 'bn', 'in', or 'ln'. Given: {norm_type}")

        self.inplanes = self.INIT_DIM
        first_out_planes = self.inplanes
        self.blocks = [
            nn.Sequential(
                ConvNormActivation(
                    in_channels, first_out_planes, kernel_size=7, stride=first_stride, D=D,
                    bias=bias, activation_layer=self.act_fn, norm_layer=self.norm_layer
                ),
                ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
            )
        ]

        for planes, layers, stride in zip(self.PLANES, self.LAYERS, self.STRIDES):
            self.blocks.append(
                self._make_layer(self.BLOCK, planes, layers, stride=stride)
            )
        self.blocks = nn.ModuleList(self.blocks)

        self.glob_avg = GLOBAL_POOL[global_pool]()  # dimension=D)
        if dropout > 0:
            self.glob_avg = nn.Sequential(
                self.glob_avg,
                ME.MinkowskiDropout(dropout),
            )

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, ME.MinkowskiBatchNorm):
            nn.init.constant_(m.bn.weight, 1)
            nn.init.constant_(m.bn.bias, 0)

        if isinstance(m, ME.MinkowskiConvolution):
            nn.init.trunc_normal_(m.kernel, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, ME.MinkowskiLinear):
            nn.init.trunc_normal_(m.linear.weight, std=.02)
            if m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, dimension=self.D,
                    dilation=1, bias=self.bias,
                ),
                self.norm_layer(planes * block.expansion),
            )
        layers = [block(
            self.inplanes, planes, self.act_fn, stride=stride, dilation=dilation, downsample=downsample,
            dimension=self.D, drop_path=self.drop_path, bias=self.bias, norm_layer=self.norm_layer
        )]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, self.act_fn, stride=1, dilation=dilation, dimension=self.D,
                drop_path=self.drop_path, bias=self.bias, norm_layer=self.norm_layer
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.glob_avg(x)
        return self.final(x)


class ResNet14_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class ResNet18_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class ResNet34_(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResNet50_(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class ResNet101_(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)


class SENet14(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 1)
    STRIDES = (1, 2, 2, 2)


class SENet17_6deep(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 1, 2, 1)
    STRIDES = (1, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    PLANES = (32, 64, 128, 256, 512, 1024)


class SENet17_5deep(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (1, 1, 1, 2, 2)
    STRIDES = (1, 2, 2, 2, 2)
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512, 1024)


class SENet18(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (2, 2, 2, 2)
    STRIDES = (1, 2, 2, 2)


class SENet34(ResNetBase):
    BLOCK = SEBasicBlock
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SENet50(ResNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 6, 3)
    STRIDES = (1, 2, 2, 2)


class SENet101(ResNetBase):
    BLOCK = SEBottleneck
    LAYERS = (3, 4, 23, 3)
    STRIDES = (1, 2, 2, 2)
