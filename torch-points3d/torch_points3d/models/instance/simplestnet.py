import logging
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from torch_points3d.models.instance.base import InstanceBase

log = logging.getLogger(__name__)


class SeparateLinear(torch.nn.Module):

    def __init__(self, in_channel, num_reg_classes):
        super(SeparateLinear, self).__init__()
        self.linears = []
        if num_reg_classes > 0:
            self.linears += [torch.nn.Linear(in_channel, 1, bias=True) for i in range(num_reg_classes)]

        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, x):
        return torch.cat([lin(x) for lin in self.linears], 1)


class SimplestNet(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        super(SimplestNet, self).__init__(option, model_type, dataset, modules)
        self.model = nn.Sequential(
            nn.Conv1d(dataset.feature_dimension + 3, 64, 1),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 1),
            nn.GELU(),
            nn.BatchNorm1d(128),
        )
        self.head = SeparateLinear(128, self.num_reg_classes)
        self.dataset_num_points = dataset.dataset_opt.fixed.num_points
        self._supports_mixed = True

        self.head_namespace = option.get("head_namespace", "head.linears")
        self.head_optim_settings = option.get("head_optim_settings", {})
        self.backbone_optim_settings = option.get("backbone_optim_settings", {})

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
        self.data_visual = data
        points = data['pos'].to(device)
        points = points.view(-1, self.dataset_num_points, points.shape[-1])

        features = data['x'].to(device)
        features = features.view(-1, self.dataset_num_points, features.shape[-1])

        self.input = torch.cat([features, points], 2).moveaxis(2, 1)
        self.batch_idx = data.batch

        if len(self.loss_fns) > 0:
            bs = len(data)
            if self.has_reg_targets and data.y_reg is not None:
                self.reg_y_mask = data.y_reg_mask.to(device).view(bs, -1)
                self.reg_y = data.y_reg.to(device).view(bs, -1)

    def compute_loss(self):
        self.loss = 0
        self.compute_instance_loss()

    def forward(self, *args, **kwargs):
        x = self.model(self.input)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        self.output = self.head(x)

        self.reg_out = self.convert_outputs(self.output)
        self.compute_loss()

        self.data_visual.pred = self.output
