import logging
from typing import List

import numpy as np
import torch
from easydict import EasyDict
from torch import nn

from torch_points3d.models.instance.base import InstanceBase

from torch_points3d.modules.KPConv.architectures import KPCNN
from torch_points3d.modules.KPConv.common import batch_grid_subsampling, batch_neighbors

from time import time

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


class KPConv(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        super(KPConv, self).__init__(option, model_type, dataset, modules)

        self.config = config = option.config

        self.model = KPCNN(config)

        self.neighborhood_limits = []
        in_channel = self.model.head_mlp.mlp.weight.shape[1]
        self.head = self.init_head(in_channel)

        self.dataset_num_points = dataset.dataset_opt.fixed.num_points
        self.model_num_points = option.get("num_points", None)
        if self.model_num_points is None:
            self.should_sample = False
        else:
            from openpoints.models.layers import furthest_point_sample
            self.furthest_point_sample = furthest_point_sample
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

        self.head_optim_settings = option.get("head_optim_settings", {})
        self.backbone_optim_settings = option.get("backbone_optim_settings", {})

    def get_parameter_list(self) -> List[dict]:
        params_list = []
        head_parameters = self.head.parameters()

        backbone_parameters = self.model.parameters()

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

        points = data['pos']
        features = data['x']

        if self.should_sample:  # point resampling strategy if same number of points
            points = points.view(-1, self.dataset_num_points, points.shape[-1]).to(device)
            features = features.view(-1, self.dataset_num_points, features.shape[-1]).to(device)
            point_all = points.size(1) if points.size(1) < self.point_all else self.point_all
            fps_idx = self.furthest_point_sample(points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(point_all, self.model_num_points, False)]
            points = torch.gather(points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
            features = torch.gather(features, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, features.shape[-1]))

        self.batch_idx = data.batch
        lengths = data.ptr[1:] - data.ptr[:-1]

        # TODO could to this in batch pre collate
        self.input = EasyDict(self.prepare_inputs(
            points.view(-1, 3).cpu().numpy(),
            features.view(-1, features.shape[-1]).cpu().numpy(),
            lengths.numpy().astype(np.int32),
            device
        ))

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

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def prepare_inputs(self, stacked_points, stacked_features, stack_lengths, device):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture
        L = 0
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue
            L += 1
            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating the farthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths
        li += [stacked_features, ]

        # Extract input tensors from the list of numpy array
        input = {}
        ind = 0
        input["points"] = [torch.from_numpy(nparray).to(device) for nparray in li[ind:ind + L]]
        ind += L
        input["neighbors"] = [torch.from_numpy(nparray).to(device) for nparray in li[ind:ind + L]]
        ind += L
        input["pools"] = [torch.from_numpy(nparray).to(device) for nparray in li[ind:ind + L]]
        ind += L
        input["lengths"] = [torch.from_numpy(nparray).to(device) for nparray in li[ind:ind + L]]
        ind += L
        input["features"] = torch.from_numpy(li[ind]).to(device)

        return input

    def compute_loss(self):
        self.loss = 0
        self.compute_instance_loss()

    def forward(self, *args, **kwargs):
        out = self.model(self.input)
        self.output = self.head(out)
        self.reg_out, self.mol_out, self.cls_out = self.convert_outputs(self.output)
        self.compute_loss()

        self.data_visual.pred = self.output
