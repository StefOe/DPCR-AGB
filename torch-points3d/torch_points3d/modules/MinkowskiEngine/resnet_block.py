# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch.nn as nn

import MinkowskiEngine as ME

from torch_points3d.modules.MinkowskiEngine.common import MinkowskiDropPath


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 act_fn,
                 norm_layer,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 drop_path=0.0,
                 bias: bool = True,
                 dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension, bias=bias,
        )
        self.norm1 = norm_layer(planes)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension, bias=bias,
        )
        self.norm2 = norm_layer(planes)
        self.relu = act_fn
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.drop_path = MinkowskiDropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        residual = self.downsample(residual)

        out = self.drop_path(out) + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 act_fn,
                 norm_layer,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 drop_path=0.0,
                 bias: bool = True,
                 dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=1, dimension=dimension, bias=bias,
        )
        self.norm1 = norm_layer(planes)

        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension, bias=bias,
        )
        self.norm2 = norm_layer(planes)

        self.conv3 = ME.MinkowskiConvolution(
            planes, planes * self.expansion, kernel_size=1, dimension=dimension, bias=bias
        )
        self.norm3 = norm_layer(planes * self.expansion)

        self.relu = act_fn
        self.downsample = downsample if downsample is not None else nn.Identity()
        self.drop_path = MinkowskiDropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        residual = self.downsample(residual)

        out = self.drop_path(out) + residual
        out = self.relu(out)

        return out
