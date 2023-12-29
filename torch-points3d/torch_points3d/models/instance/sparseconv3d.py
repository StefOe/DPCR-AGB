import logging

from torch import nn
import torch

from torch_points3d.models.instance.base import InstanceBase
from torch_points3d.models.regression.minkowski import SeparateLinear
from torch_points3d.modules.SparseConv3d.SENet import ResNetBase, NETWORK_CONFIGS

log = logging.getLogger(__name__)


class SeparateLinear(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(SeparateLinear, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_channel, 1, bias=True) for i in range(out_channels)])

    def forward(self, x):
        return torch.cat([lin(x) for lin in self.linears], 1)


class ResNetModel(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        # call the initialization method
        super().__init__(option, model_type, dataset, modules)
        self.model = ResNetBase(
            dataset.feature_dimension, dataset.num_classes, activation=option.activation,
            first_stride=option.first_stride, dropout=option.dropout, global_pool=option.global_pool,
            backend=option.backend, **NETWORK_CONFIGS[option.model_name])

        in_channel = self.model.final.weight.shape[1]
        out_channel = self.model.final.weight.shape[0]
        self.model.final = SeparateLinear(in_channel, out_channel)
        self._supports_mixed = self.model.snn.name == "torchsparse"

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        self.input = self.model.snn.SparseTensor(data.x, data.coords, data.batch, device)
        if data.y is not None:
            self.labels = data.y.to(device)
        else:
            self.labels = None

        self.data_visual = data

    def compute_loss(self):
        self.loss_regr = 0
        labels = self.labels.view(self.output.shape)
        for loss_fn in self.loss_fns:
            self.loss_regr += (loss_fn(self.output, labels, reduction="none") / self.scale_targets).mean()

        self.loss_regr += self.get_internal_loss()
        self.loss = self.loss_regr

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
        if self.labels is not None:
            self.compute_loss()

        self.data_visual.pred = self.output
