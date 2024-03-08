import logging
from typing import List

import numpy as np
import torch
from torch import nn
from torch_points3d.models.instance.base import InstanceBase


log = logging.getLogger(__name__)


class SeparateLinear(torch.nn.Module):

    def __init__(self, in_channel, out_channels):
        super(SeparateLinear, self).__init__()
        if isinstance(out_channels, int):
            self.linears = nn.ModuleList([nn.Linear(in_channel, 1, bias=True) for i in range(out_channels)])
        elif isinstance(out_channels, dict):
            num_reg_classes = out_channels.get("num_reg_classes", 0)

            self.linears = []
            if num_reg_classes > 0:
                self.linears += [torch.nn.Linear(in_channel, 1, bias=True) for i in range(num_reg_classes)]

            self.linears = torch.nn.ModuleList(self.linears)
        else:
            self.linears = nn.ModuleList([nn.Linear(in_channel, 1, bias=True)])

    def forward(self, x):
        return torch.cat([lin(x) for lin in self.linears], 1)


class PointNext(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        super(PointNext, self).__init__(option, model_type, dataset, modules)
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig
        from openpoints.models.layers import furthest_point_sample
        self.furthest_point_sample = furthest_point_sample
        stride = option.stride
        use_mlps = option.get("use_mlps", True)
        radius_scaling = option.get("radius_scaling", 2.)
        radius = option.get("radius", 0.1)
        nsample = option.get("nsample", 32)
        act = option.get("activation", "relu")
        act_args = EasyConfig({'act': act})
        if act in ["elu", "celu"]:
            act_args["alpha"] = 0.54

        MODEL = {
            "pointnet": EasyConfig({
                'NAME': 'BaseCls',
                'encoder_args': EasyConfig({
                    'NAME': 'PointNetEncoder',
                    'in_channels': dataset.feature_dimension,
                    'is_seg': False,
                    'input_transform': False,
                }),
                'cls_args': EasyConfig({
                    'NAME': 'ClsHead',
                    'num_classes': dataset.num_classes,
                    'act_args': act_args,
                    'mlps': [512, 256, 128, 128] if use_mlps else [],
                })
            }),
            "pointnext_s": EasyConfig({
                'NAME': 'BaseCls',
                'encoder_args': EasyConfig({
                    'NAME': 'PointNextEncoder',
                    "blocks": [1, 1, 1, 1, 1, 1],
                    'strides': [1, stride, stride, stride, stride, 1],
                    'width': 32,
                    'in_channels': dataset.feature_dimension,
                    'radius': radius,
                    'radius_scaling': radius_scaling,
                    'sa_layers': 2,
                    'sa_use_res': True,
                    'nsample': nsample,
                    'expansion': 4,
                    'aggr_args': EasyConfig({'feature_type': 'dp_fj', 'reduction': 'max'}),
                    'group_args': EasyConfig({'NAME': 'ballquery', 'normalize_dp': True}),
                    'conv_args': EasyConfig({'order': 'conv-norm-act'}),
                    'act_args': act_args,
                    'norm_args': EasyConfig({'norm': 'bn'})
                }),
                'cls_args': EasyConfig({
                    'NAME': 'ClsHead',
                    'num_classes': dataset.num_classes,
                    'act_args': act_args,
                    'mlps': [512, 256] if use_mlps else [],
                    'norm_args': EasyConfig({'norm': 'bn1d'})
                })
            }),
            "pointnext_b": EasyConfig({
                'NAME': 'BaseCls',
                'encoder_args': EasyConfig({
                    'NAME': 'PointNextEncoder',
                    'blocks': [1, 2, 3, 2, 1, 1],
                    'strides': [1, stride, stride, stride, stride, 1],
                    'width': 32,
                    'in_channels': dataset.feature_dimension,
                    'radius': radius,
                    'radius_scaling': radius_scaling,
                    'sa_layers': 1,
                    'sa_use_res': False,
                    'nsample': nsample,
                    'expansion': 4,
                    'aggr_args': EasyConfig({'feature_type': 'dp_fj', 'reduction': 'max'}),
                    'group_args': EasyConfig({'NAME': 'ballquery', 'normalize_dp': True}),
                    'conv_args': EasyConfig({'order': 'conv-norm-act'}),
                    'act_args': act_args,
                    'norm_args': EasyConfig({'norm': 'bn'})
                }),
                'cls_args': EasyConfig({
                    'NAME': 'ClsHead',
                    'num_classes': dataset.num_classes,
                    'act_args': act_args,
                    'mlps': [512, 256] if use_mlps else [],
                    'norm_args': EasyConfig({'norm': 'bn1d'})
                })
            })
        }

        cfg = MODEL[option.arch]

        cfg = EasyConfig(cfg)
        self.model = build_model_from_cfg(cfg)

        in_channel = self.model.prediction.head[-1][0].weight.shape[1]
        self.model.prediction.head[-1] = self.init_head(in_channel)

        self.dataset_num_points = dataset.dataset_opt.fixed.num_points
        self.model_num_points = option.num_points
        if self.model_num_points == 1024:
            self.point_all = 1200
        elif self.model_num_points == 4096:
            self.point_all = 4800
        elif self.model_num_points == 6144:
            self.point_all = 6900
        elif self.model_num_points == 8192:
            self.point_all = 8192
        elif self.model_num_points == 12288:
            self.point_all = 12288
        elif self.model_num_points == 16384:
            self.point_all = 16384
        else:
            raise NotImplementedError()
        self.should_sample = self.model_num_points < self.dataset_num_points
        self._supports_mixed = True

        self.head_namespace = option.get("head_namespace", "linears")
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

    def init_head(self, in_channel):
        return SeparateLinear(
            in_channel, {
                "num_reg_classes": self.num_reg_classes,
            }
        )

    def set_input(self, data, device):
        self.data_visual = data

        points = data['pos'].to(device)
        points = points.view(-1, self.dataset_num_points, points.shape[-1])

        features = data['x'].to(device)
        features = features.view(-1, self.dataset_num_points, features.shape[-1])

        # # debug
        # from openpoints.dataset import vis_points
        # import ipdb; ipdb.set_trace()
        # vis_points(data['pos'])

        if self.should_sample:  # point resampling strategy
            point_all = points.size(1) if points.size(1) < self.point_all else self.point_all
            fps_idx = self.furthest_point_sample(points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, self.model_num_points, False)]
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
            features = torch.gather(features, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, features.shape[-1]))

        self.input = {"pos": points, "x": features.transpose(1, 2).contiguous()}
        self.batch_idx = data.batch

        if len(self.loss_fns) > 0:
            bs = len(data)
            if self.has_reg_targets and data.y_reg is not None:
                self.reg_y_mask = data.y_reg_mask.to(device).view(bs, -1)
                self.reg_y = data.y_reg.to(device).view(bs, -1)
            if self.has_mol_targets and data.y_mol is not None:
                self.mol_y_mask = data.y_mol_mask.to(device).view(bs, -1)
                self.mol_y = data.y_mol.to(device).view(bs, -1)
            if self.has_cls_targets and data.y_cls is not None:
                self.cls_y_mask = data.y_cls_mask.to(device).view(bs, -1)
                self.cls_y = data.y_cls.to(device).view(bs, -1)

    def compute_loss(self):
        self.loss = 0
        self.compute_instance_loss()

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
        self.reg_out = self.convert_outputs(self.output)
        self.compute_loss()

        self.data_visual.pred = self.output