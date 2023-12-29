import collections
import random
from enum import Enum
from functools import partial

import MinkowskiEngine as ME
import torch
import torch.nn as nn
from MinkowskiEngine import MinkowskiNonlinearity as NL
from MinkowskiEngine import SparseTensor


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels), ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
        )
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


ACTIVATIONS = {
    "relu": partial(NL.MinkowskiReLU, inplace=True),
    "celu": partial(NL.MinkowskiCELU, inplace=True, alpha=0.54),
    "silu": partial(NL.MinkowskiSiLU, inplace=True),
    "swish": partial(NL.MinkowskiSiLU, inplace=True),
    "elu": partial(NL.MinkowskiELU, inplace=True, alpha=0.54),
    "sigmoid": partial(NL.MinkowskiSigmoid),
    "tanh": partial(NL.MinkowskiTanh),
    "siren": partial(NL.MinkowskiSinusoidal),
    "gelu": partial(NL.MinkowskiGELU),
}

GLOBAL_POOL = {
    "max": ME.MinkowskiGlobalMaxPooling,
    "mean": ME.MinkowskiGlobalAvgPooling,
    "sum": ME.MinkowskiGlobalSumPooling,
}


class ConvType(Enum):
    """
  Define the kernel region type
  """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = 6, "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS "

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Covert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.CUSTOM,
}

int_to_region_type = {0: ME.RegionType.HYPER_CUBE, 1: ME.RegionType.HYPER_CROSS, 2: ME.RegionType.CUSTOM}


def convert_region_type(region_type):
    """
  Convert the integer region_type to the corresponding RegionType enum object.
  """
    return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size, ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, collections.Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [kernel_size, ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims and CROSS conv for temp dim
        if D < 4:
            region_type = ME.RegionType.HYPER_CUBE
        else:
            axis_types = [ME.RegionType.HYPER_CUBE, ] * 3
            if D == 4:
                axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(in_planes, out_planes, kernel_size, stride=1, dilation=1, bias=False, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D
    )

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def conv_tr(
        in_planes, out_planes, kernel_size, upsample_stride=1, dilation=1, bias=False, conv_type=ConvType.HYPERCUBE,
        D=-1
):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, upsample_stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D
    )

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D,
    )


def avg_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, in_coords_key=None, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D
    )

    return ME.MinkowskiAvgPooling(
        kernel_size=kernel_size, stride=stride, dilation=dilation, kernel_generator=kernel_generator, dimension=D
    )


def avg_unpool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D
    )

    return ME.MinkowskiAvgUnpooling(
        kernel_size=kernel_size, stride=stride, dilation=dilation, kernel_generator=kernel_generator, dimension=D
    )


def sum_pool(kernel_size, stride=1, dilation=1, conv_type=ConvType.HYPERCUBE, D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size, stride, dilation, region_type=region_type, axis_types=axis_types, dimension=D
    )

    return ME.MinkowskiSumPooling(
        kernel_size=kernel_size, stride=stride, dilation=dilation, kernel_generator=kernel_generator, dimension=D
    )


class ConvNormActivation(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride, norm_layer,
                 activation_layer, bias, D):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            input_channels, out_channels, kernel_size=kernel_size, stride=stride, dimension=D, bias=bias
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.Identity() if activation_layer is None else activation_layer

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def batch_norm(X, moving_mean, moving_var, gamma, beta, training, momentum, eps, meanpool):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled() or not training:
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X.F - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)

        # When using a fully connected layer, calculate the mean and
        # variance on the feature dimension
        mean = meanpool(X).F.mean(0)
        diff = X.F - mean
        var = (diff ** 2).mean(0)

        # In training mode, the current mean and variance are used
        X_hat = diff / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        if training:
            moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
            moving_var = (1.0 - momentum) * moving_var + momentum * var
    return gamma * X_hat + beta  # Scale and shift


class MinkowskiBatchNorm(nn.BatchNorm1d):
    r"""A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(MinkowskiBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.meanpool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, input_):
        input = input_.F
        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        output = batch_norm(
            input_,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            self.momentum,
            self.eps,
            self.meanpool
        )

        return SparseTensor(
            output,
            coordinate_map_key=input_.coordinate_map_key,
            coordinate_manager=input_.coordinate_manager,
        )

    def __repr__(self):
        s = "({}, eps={}, momentum={}, affine={}, track_running_stats={})".format(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats,
        )
        return self.__class__.__name__ + s


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
            self.gamma * (x.F * Nx) + self.beta + x.F,
            coordinate_map_key=in_key,
            coordinate_manager=cm
        )


class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
            x.F * mask,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)


class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """

    def __init__(
            self,
            normalized_shape,
            eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)
