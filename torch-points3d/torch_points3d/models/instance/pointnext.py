import logging
from typing import List

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch

from openpoints.models.layers import create_act
from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.models.instance.base import InstanceBase
from torch_points3d.models.instance.semi_supervised_helper import invariance_loss, gather, variance_loss, \
    covariance_loss, barlow_loss

log = logging.getLogger(__name__)


class SeparateLinear(torch.nn.Module):

    def __init__(self, in_channel, out_channels):
        super(SeparateLinear, self).__init__()
        if isinstance(out_channels, int):
            self.linears = nn.ModuleList([nn.Linear(in_channel, 1, bias=True) for i in range(out_channels)])
        elif isinstance(out_channels, dict):
            num_reg_classes = out_channels.get("num_reg_classes", 0)
            num_mixtures = out_channels.get("num_mixtures", [])
            num_cls_classes = out_channels.get("num_cls_classes", [])

            self.linears = []
            if num_reg_classes > 0:
                self.linears += [torch.nn.Linear(in_channel, 1, bias=True) for i in range(num_reg_classes)]
            if len(num_mixtures) > 0:
                self.linears += [
                    torch.nn.Linear(in_channel, num_mixtures * 3, bias=True) for i, num_mixtures in
                    enumerate(num_mixtures)
                ]
            if len(num_cls_classes) > 0:
                self.linears += [
                    torch.nn.Linear(in_channel, num_classes) for num_classes in num_cls_classes
                ]

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
                "num_mixtures": self.num_mixtures,
                "num_cls_classes": self.num_cls_classes
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
        self.reg_out, self.mol_out, self.cls_out = self.convert_outputs(self.output)
        self.compute_loss()

        self.data_visual.pred = self.output


class ProjClassifier(nn.Module):
    def __init__(self, hidden_dim: int, proj_layers, out_channels: [int, dict], detach_classifier: bool, act_fn,
                 last_norm: bool):
        nn.Module.__init__(self)
        sizes = [hidden_dim] + list(proj_layers)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(FastBatchNorm1d(sizes[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if last_norm:
            layers.append(FastBatchNorm1d(sizes[-1], affine=False))
        self.projector = nn.Sequential(*layers)
        self.detach_classifier = detach_classifier

        self.classifier = SeparateLinear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor):
        x_ = x.detach() if self.detach_classifier else x
        return self.classifier(x_), self.projector(x)


class PointNextBarlowTwin(PointNext):
    def __init__(self, option, model_type, dataset, modules):
        model_version = option.get("model_version", "standard")
        self.proj_layers = option.proj_layers
        self.proj_last_norm = option.proj_last_norm
        self.proj_activation = option.get("proj_activation", None)
        if self.proj_activation is None:
            self.proj_activation = option.get("activation", "relu")
        self.detach_classifier = option.mode != "finetune" and model_version == "standard"
        self.reset_output = option.get("reset_output", True)

        super().__init__(option, model_type, dataset, modules)

        self.mode = option.mode
        if self.mode not in ["finetune", "freeze"]:
            self.loss_names.extend(
                ["loss_self_supervised"]
            )
        self.scale_loss = option.scale_loss
        self.backbone_lr = option.backbone_lr

    def init_head(self, in_channel):
        self.act_fn = create_act(self.proj_activation)
        return ProjClassifier(
            in_channel, self.proj_layers,
            {
                "num_reg_classes": self.num_reg_classes,
                "num_mixtures": self.num_mixtures,
                "num_cls_classes": self.num_cls_classes
            }, self.detach_classifier, self.act_fn, self.proj_last_norm
        )

    def get_parameter_list(self) -> List[dict]:
        params_list = []
        classifier_parameters, model_parameters = [], []
        for name, param in self.model.named_parameters():
            if "prediction.head" in name:
                classifier_parameters.append(param)
            else:
                model_parameters.append(param)

        params_list.append({"params": classifier_parameters})
        if self.mode in ["finetune", "train"]:
            model_dict = {"params": model_parameters}
            if self.backbone_lr != "base_lr":
                model_dict["lr"] = self.backbone_lr
            params_list.append(model_dict)

        return params_list

    def set_pretrained_weights(self):
        super().set_pretrained_weights()
        if self.mode in ["finetune", "freeze"] and self.reset_output:
            log.info(f"resetting weights for final prediction layer (since we are in {self.mode} mode)")
            for m in self.model.prediction.head[-1].classifier.linears:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def set_input(self, data, device):

        points = data['pos'].to(device)
        points = points.view(-1, self.dataset_num_points, points.shape[-1])

        features = data['x'].to(device)
        features = features.view(-1, self.dataset_num_points, features.shape[-1])

        # # debug
        # from openpoints.vis3d import vis_points
        # vis_points(data['pos'].cpu().numpy()[0])
        # import ipdb; ipdb.set_trace()

        if self.should_sample:  # point resampling strategy
            point_all = points.size(1) if points.size(1) < self.point_all else self.point_all
            fps_idx = self.furthest_point_sample(points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, self.model_num_points, False)]
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
            features = torch.gather(features, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, features.shape[-1]))

        self.batch_idx = data.batch
        if self.training and self.double_batch:
            self.input = {"pos": points[::2].contiguous(), "x": features[::2].transpose(1, 2).contiguous()}
            self.input2 = {"pos": points[1::2].contiguous(), "x": features[1::2].transpose(1, 2).contiguous()}
            data = Batch.from_data_list(data.to_data_list()[::2])
        else:
            self.input = {"pos": points, "x": features.transpose(1, 2).contiguous()}
            self.input2 = None

        self.data_visual = data

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

        if self.mode not in ["finetune", "freeze"]:
            self.compute_self_supervised_loss()

    def compute_self_supervised_loss(self):
        # barlow loss
        # empirical cross-correlation matrix
        self.loss_self_supervised = 0
        if self.training and self.double_batch:
            self.loss_self_supervised += barlow_loss(
                self.z1, self.z2, self.scale_loss["lambda"]
            )
            self.loss += self.scale_loss["all"] * self.loss_self_supervised

    def compute_instance_loss(self):
        self.compute_reg_loss()
        self.compute_mol_loss()
        self.compute_cls_loss()

    def forward_(self, input1, input2):
        class_out_1, z1 = self.model(input1)
        if self.training and self.mode == "train":
            class_out_2, z2 = self.model(input2)
        else:
            class_out_2, z2 = None, None

        return class_out_1, class_out_2, z1, z2

    def forward(self, *args, **kwargs):
        self.set_mode()
        self.output, self.output2, self.z1, self.z2 = self.forward_(self.input, self.input2)
        self.reg_out, self.mol_out, self.cls_out = self.convert_outputs(self.output)
        self.reg_out2, self.mol_out2, self.cls_out2 = self.convert_outputs(self.output2)

        self.compute_loss()
        self.data_visual.pred = self.output

    def set_mode(self):
        if self.training:
            if self.mode == "freeze":
                self.model.requires_grad_(False)
                self.model.prediction.head[-1].requires_grad_(True)
                self.model.eval()
                self.model.prediction.head[-1].train()


class PointNextVICReg(PointNextBarlowTwin):

    def __init__(self, option, model_type, dataset, modules):
        super(PointNextVICReg, self).__init__(option, model_type, dataset, modules)

        if self.mode not in ["finetune", "freeze"]:
            self.loss_names.extend(
                ["loss_invariance", "loss_variance", "loss_covariance"]
            )

    def compute_self_supervised_loss(self):
        # barlow loss
        # empirical cross-correlation matrix
        self.loss_self_supervised = 0
        if self.training and self.mode == "train":
            # from https://github.com/vturrisi/solo-learn/blob/6f19d5dc38fb6521e7fdd6aed5ac4a30ef8f3bd8/solo/losses/vicreg.py#L83
            z1, z2 = self.z1, self.z2
            # invariance loss
            self.loss_invariance = invariance_loss(z1, z2)

            # vicreg's official code gathers the tensors here
            # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
            z1, z2 = gather(z1), gather(z2)

            # variance_loss
            self.loss_variance = variance_loss(z1, z2)
            self.loss_covariance = covariance_loss(z1, z2)
            loss = self.scale_loss["invariance"] * self.loss_invariance + \
                   self.scale_loss["variance"] * self.loss_variance + \
                   self.scale_loss["covariance"] * self.loss_covariance

            self.loss_self_supervised += loss
            self.loss += self.loss_self_supervised
