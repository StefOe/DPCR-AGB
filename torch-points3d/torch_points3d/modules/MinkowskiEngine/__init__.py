import sys


from .networks import *
from .SENet import *
from .VAE import *
from .barlow import *
from .UNet import *
from .res16unet import *
from .resunet import *
from .PointNet import MinkowskiPointNet

_custom_models = sys.modules[__name__]


def initialize_minkowski_unet(
        model_name, in_channels, out_channels, D=3, conv1_kernel_size=3, **kwargs
):
    net_cls = getattr(_custom_models, model_name)
    return net_cls(
        in_channels=in_channels, out_channels=out_channels, D=D, conv1_kernel_size=conv1_kernel_size, **kwargs
    )



