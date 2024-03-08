import logging
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch_points3d.models.base_architectures import BackboneBasedModel
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.model_interface import InstanceTrackerInterface

log = logging.getLogger(__name__)


def mape(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    mask = y != 0
    error = torch.zeros_like(y)
    error[mask] = torch.abs((y[mask] - x[mask]) / y[mask])

    if reduce:
        return error.mean()
    else:
        return error


def smape(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    error = ((y - x).abs() / (torch.abs(x) + torch.abs(y) + torch.finfo(torch.float16).eps))

    if reduce:
        return error.mean()
    else:
        return error

REG_LOSSES = {
    "smoothl1": F.smooth_l1_loss,
    "l2": F.mse_loss,
    "l1": F.l1_loss,
    "mape": mape,
    "smape": smape,
}


def linear(x): return x


OUT_ACT = {
    "linear": linear,
    "elu": partial(F.elu, inplace=True),
    "relu": partial(F.relu, inplace=True),
}


class InstanceBase(BaseModel, InstanceTrackerInterface):
    def __init__(self, option, model_type, dataset, modules):
        super().__init__(option)
        self.visual_names = ["data_visual"]

        self.loss_fns = {}
        self.has_reg_targets = dataset.has_reg_targets
        self.reg_targets_idx = dataset.reg_targets_idx

        if self.has_reg_targets:
            self.loss_names.append("loss_reg")

            self.get_task_weights_scale_center(
                dataset, task="regression", short_task="reg", default_norm="standard", targets_idx=self.reg_targets_idx
            )

            self.reg_out_act = OUT_ACT[option.get("reg_out_activation", "linear").lower()]
            self.reg_report_out_act = OUT_ACT[option.get("reg_out_report_activation", "linear").lower()]

            loss_strs = option.get("reg_loss_fn", "smoothl1")
            if len(loss_strs) > 0:
                loss_strs = loss_strs.split(",")
                self.loss_fns["reg"] = []
                for loss_str in loss_strs:
                    loss = REG_LOSSES[loss_str]
                    self.loss_fns["reg"].append(loss)

        self.num_reg_classes = dataset.num_reg_classes

        # model overrides dataset settings
        self.double_batch = option.get("double_batch", dataset.double_batch)

    def get_task_weights_scale_center(self, dataset, task, short_task, default_norm, targets_idx):
        center = np.zeros(sum(targets_idx))
        scale = np.ones(sum(targets_idx))
        i = 0
        weights = []
        for target in dataset.targets:
            if dataset.targets[target]["task"] == task:
                weights.append(dataset.targets[target].get("weight", 1))
                normalization = dataset.targets[target].get("normalization", default_norm)
                idx = np.zeros_like(targets_idx)
                idx[i] = True
                if normalization == "standard":
                    center[i] = self.get_dataset_avg_stat(dataset, "mean", default=0.0, feat_idx=idx)
                    scale[i] = self.get_dataset_avg_stat(dataset, "std", default=1.0, feat_idx=idx)
                elif normalization == "min-max":
                    center[i] = self.get_dataset_avg_stat(dataset, "min", default=0.0, feat_idx=idx)
                    scale[i] = self.get_dataset_avg_stat(dataset, "max", default=1.0, feat_idx=idx) - center[i]
                else:
                    if normalization != "none":
                        log.warning(f"'{normalization}' is not a valid normalization, using no normalization")

                center[i] = dataset.targets[target].get("center_override", center[i])
                scale[i] = dataset.targets[target].get("scale_override", scale[i])

                scale[i] *= dataset.targets[target].get("scale_mult", 1.)
                i += 1
        self.register_buffer(f"{short_task}_scale_targets", torch.tensor(scale.reshape(1, -1), dtype=torch.float))
        self.register_buffer(f"{short_task}_center_targets", torch.tensor(center.reshape(1, -1), dtype=torch.float))
        self.register_buffer(f"{short_task}_weights", torch.tensor(weights, dtype=torch.float))

    def get_dataset_avg_stat(self, dataset, stat, default, feat_idx):
        value = np.array([
            area["train"][feat_idx] for area in
            getattr(dataset, f"get_{stat}_targets")().values() if "train" in area
        ])
        nans = np.isnan(value)

        if nans.all(0).any():
            value = np.array([default] * len(feat_idx))
            log.warning(f"All training area with no valid {stat} value, setting to {default}. "
                        "This is fine if reloading amodel overrides this.")
            return value
        elif nans.all(0).any():
            idx = np.argwhere(nans.all(0))
            value[:, idx] = default
            log.warning(f"Some training area with no valid {stat} value, setting the missing to {default}. "
                        "This is fine if reloading amodel overrides this.")

        return np.nanmean(value, 0)

    def set_input(self, data, device):
        raise NotImplemented

    def convert_outputs(self, outputs):
        reg_out = None
        if outputs is not None:

            if self.has_reg_targets:
                reg_out = self.reg_out_act(outputs[:, :self.num_reg_classes])

        return reg_out

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplemented

    def compute_loss(self):
        raise NotImplemented

    def compute_reg_loss(self):
        if self.has_reg_targets and len(self.loss_fns["reg"]) > 0 and self.reg_y_mask.any():
            self.loss_reg = 0
            # scaling by std to have equal grads
            output = self.reg_out
            labels = ((self.reg_y - self.reg_center_targets) / self.reg_scale_targets)
            if self.training and self.double_batch:
                output2 = self.reg_out2

            if not self.reg_y_mask.all():
                output = output[self.reg_y_mask]
                labels = labels[self.reg_y_mask]
                if self.training and self.double_batch:
                    output2 = output2[self.reg_y_mask]

            for loss_fn in self.loss_fns["reg"]:

                if self.training and self.double_batch:
                    self.loss_reg += (
                            (0.5 * loss_fn(output, labels, reduce=False)) +
                            (0.5 * loss_fn(output2, labels, reduce=False))
                    ).mean()
                else:
                    self.loss_reg += loss_fn(output, labels, reduce=True)

            self.loss += self.reg_weights.mean() * self.loss_reg

    def get_reg_output(self):
        """ returns a tensor of size ``[N_points,N_regression_targets]`` where each value is the regression output
        of the network for a point (output of the last layer in general)
        """
        return self.reg_report_out_act(self.reg_out * self.reg_scale_targets + self.reg_center_targets)

    def get_reg_input(self):
        """ returns the last regression input that was given to the model or raises error
        """
        return self.reg_y

    def compute_instance_loss(self):
        self.compute_reg_loss()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if not self.opt.get("override_target_stats", True):
            remove_dict_entry(state_dict, "reg_scale_targets")
            remove_dict_entry(state_dict, "reg_center_targets")
            remove_dict_entry(state_dict, "reg_weights")

        super().load_state_dict(state_dict, strict)


def remove_dict_entry(dict, key):
    if key in dict:
        del dict[key]
        log.info(f"removed '{key}', will use dataset value instead")
    return dict


class InstanceBackboneBasedModel(BackboneBasedModel):
    def __init__(self, option, model_type, dataset, modules_lib):
        super().__init__(option, model_type, dataset, modules_lib)
        self.visual_names = ["data_visual"]

        self.loss_fns = {}
        self.has_reg_targets = dataset.has_reg_targets

        if self.has_reg_targets:

            if dataset.has_reg_targets:
                self.loss_names.append("loss_reg")
                scale = np.nanmean([area["train"] for area in dataset.get_std_targets().values()])
                self.register_buffer("reg_scale_targets", torch.tensor(scale.reshape(1, -1), dtype=torch.float))
                loss_strs = option.get("reg_loss_fn", "smoothl1")
                if len(loss_strs) > 0:
                    loss_strs = loss_strs.split(",")
                    for loss_str in loss_strs:
                        loss = REG_LOSSES[loss_str]
                        self.loss_fns["reg"].append(loss)

    def set_input(self, data, device):
        raise NotImplemented

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplemented

    def compute_loss(self):
        raise NotImplemented

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G


class Instance_MP(InstanceBackboneBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        super().__init__(option, model_type, dataset, modules)  # call the initialization method of RegressionBase

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get("dropout")
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[4], dataset.num_classes)

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch

    def compute_loss(self):
        self.loss_regr = 0
        labels = self.labels.view(self.output.shape)
        for loss_fn in self.loss_fns:
            self.loss_regr += loss_fn(self.output, labels)

        self.loss = self.loss_regr

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data = self.down_modules[0](self.input)

        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin3(x)
        self.output = x

        if self.labels is not None:
            self.compute_loss()

        self.data_visual = self.input
        self.data_visual.y = self.labels
        self.data_visual.pred = self.output
        return self.output
