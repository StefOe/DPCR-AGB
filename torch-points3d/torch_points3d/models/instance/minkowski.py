import logging
from typing import List

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from torch_points3d.models.instance.base import InstanceBase
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

log = logging.getLogger(__name__)


class SeparateLinear(torch.nn.Module):

    def __init__(self, in_channel, num_reg_classes):
        super(SeparateLinear, self).__init__()
        self.linears = []
        if num_reg_classes > 0:
            self.linears += [torch.nn.Linear(in_channel, 1, bias=True) for i in range(num_reg_classes)]

        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, x):
        return torch.cat([lin(x.F) for lin in self.linears], 1)


class MinkowskiBaselineModel(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        super(MinkowskiBaselineModel, self).__init__(option, model_type, dataset, modules)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, activation=option.activation,
            first_stride=option.first_stride, global_pool=option.global_pool, bias=option.get("bias", True),
            bn_momentum=option.get("bn_momentum", 0.1), norm_type=option.get("norm_type", "bn"),
            dropout=option.get("dropout", 0.0), drop_path=option.get("drop_path", 0.0),
            **option.get("extra_options", {})
        )
        in_channel = self.model.final.linear.weight.shape[1]
        self._supports_mixed = True
        self.model.final = SeparateLinear(in_channel, self.num_reg_classes)

        for m in self.model.final.linears:
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        self.head_namespace = option.get("head_namespace", "final.linears")
        self.head_optim_settings = option.get("head_optim_settings", {})
        self.backbone_optim_settings = option.get("backbone_optim_settings", {})

        self.add_pos = option.get("add_pos", False)

    def get_parameter_list(self) -> List[dict]:
        params_list = []
        head_parameters, backbone_parameters = [], []
        for name, param in self.model.named_parameters():
            if self.head_namespace in name:
                head_parameters.append(param)
            else:
                backbone_parameters.append(param)
        params_list.append({"params": head_parameters, **self.head_optim_settings})
        params_list.append({"params": backbone_parameters, **self.backbone_optim_settings})

        return params_list

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.data_visual = data
        features = data.x
        if self.add_pos:
            features = torch.cat([data.pos, features], 1)
        self.input = ME.SparseTensor(features=features, coordinates=coords, device=device)

        if len(self.loss_fns) > 0:
            bs = len(data)
            if self.has_reg_targets and data.y_reg is not None:
                self.reg_y_mask = data.y_reg_mask.to(device).view(bs, -1)
                self.reg_y = data.y_reg.to(device).view(bs, -1)

    def compute_loss(self):
        self.loss = 0
        self.compute_instance_loss()

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
        self.reg_out= self.convert_outputs(self.output)
        self.compute_loss()