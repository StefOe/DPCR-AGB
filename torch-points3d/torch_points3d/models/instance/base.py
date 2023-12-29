import logging
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch_points3d.core.losses.focal_loss import focal_ce
from torch_points3d.core.losses.mixture_losses import discretized_mix_logistic_loss, to_one_hot, mix_gaussian_loss
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


def smoothl1_zero(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(x[~mask], y[~mask], reduction="none")

    if reduce:
        return error.mean()
    else:
        return error


def smoothl1_zero10(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(x[mask], -torch.ones_like(y)[mask] * 10, reduction="none")
    error[~mask] = F.smooth_l1_loss(x[~mask], y[~mask], reduction="none")

    if reduce:
        return error.mean()
    else:
        return error


def smoothl1_zero_db(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    diff = F.relu(y - aug_x)
    error_aug = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    diff = F.relu(ori_x - aug_x)
    error_augx = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    loss = 2 * error + .5 * error_aug + .5 * error_augx

    if reduce:
        return loss.mean()
    else:
        return loss


def smoothl1_zero_db5(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    t = 0.01
    diff_aug = F.relu(y + t - aug_x)
    error_aug = torch.where(diff_aug < beta, 0.5 * diff_aug ** 2 / beta, diff_aug - 0.5 * beta)

    diff_augx = F.relu(ori_x + t - aug_x)
    error_augx = torch.where(diff_augx < beta, 0.5 * diff_augx ** 2 / beta, diff_augx - 0.5 * beta)

    if reduce:
        loss = 2 * error.mean() + .5 * error_aug.sum() / ((diff_aug != 0).sum() + torch.finfo(torch.float16).eps) \
               + .5 * error_augx.sum() / ((diff_augx != 0).sum() + torch.finfo(torch.float16).eps)
    else:
        loss = 2 * error + .5 * error_aug + .5 * error_augx

    return loss


def smoothl1_zero_db4(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    diff_aug = F.relu(y - aug_x)
    error_aug = torch.where(diff_aug < beta, 0.5 * diff_aug ** 2 / beta, diff_aug - 0.5 * beta)

    diff_augx = F.relu(ori_x - aug_x)
    error_augx = torch.where(diff_augx < beta, 0.5 * diff_augx ** 2 / beta, diff_augx - 0.5 * beta)

    if reduce:
        loss = 2 * error.mean() + .5 * error_aug.sum() / ((diff_aug != 0).sum() + torch.finfo(torch.float16).eps) \
               + .5 * error_augx.sum() / ((diff_augx != 0).sum() + torch.finfo(torch.float16).eps)
    else:
        loss = 2 * error + .5 * error_aug + .5 * error_augx

    return loss


def smoothl1_zero_db6(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    t = 0.00001
    diff_aug = F.relu(y + t - aug_x)
    error_aug = torch.where(diff_aug < beta, 0.5 * diff_aug ** 2 / beta, diff_aug - 0.5 * beta)

    diff_augx = F.relu(ori_x + t - aug_x)
    error_augx = torch.where(diff_augx < beta, 0.5 * diff_augx ** 2 / beta, diff_augx - 0.5 * beta)

    loss = 2 * error + .5 * error_aug + .5 * error_augx
    if reduce:
        loss = loss.sum()

    return loss


def smoothl1_zero_db7(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    diff_aug = F.relu(y - aug_x)
    error_aug = torch.where(diff_aug < beta, 0.5 * diff_aug ** 2 / beta, diff_aug - 0.5 * beta)

    diff_augx = F.relu(ori_x - aug_x)
    error_augx = torch.where(diff_augx < beta, 0.5 * diff_augx ** 2 / beta, diff_augx - 0.5 * beta)

    loss = 2 * error + .5 * error_aug + .5 * error_augx
    if reduce:
        loss = loss.sum()

    return loss


def smoothl1_zero_db3(x: torch.Tensor, y: torch.Tensor, reduce: bool = True):
    ori_x = x[::2]

    y = y[::2]
    aug_x = x[1::2]

    mask = y == 0
    error = torch.zeros_like(y)
    # replace 0 with -1
    error[mask] = F.smooth_l1_loss(ori_x[mask], -torch.ones_like(y)[mask], reduction="none")
    error[~mask] = F.smooth_l1_loss(ori_x[~mask], y[~mask], reduction="none")

    # huber loss
    beta = 1.0
    t = 0.01  # minimal increase to
    diff = F.relu(y + t - aug_x)
    error_aug = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    diff = F.relu(ori_x + t - aug_x)
    error_augx = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    loss = 2 * error + .5 * error_aug + .5 * error_augx

    if reduce:
        return loss.mean()
    else:
        return loss


REG_LOSSES = {
    "smoothl1": F.smooth_l1_loss,
    "smoothl1_zero": smoothl1_zero,
    "smoothl1_zero10": smoothl1_zero10,
    "smoothl1_zero_db": smoothl1_zero_db,
    "smoothl1_zero_db3": smoothl1_zero_db3,
    "smoothl1_zero_db4": smoothl1_zero_db4,
    "smoothl1_zero_db5": smoothl1_zero_db5,
    "smoothl1_zero_db6": smoothl1_zero_db6,
    "smoothl1_zero_db7": smoothl1_zero_db7,
    "l2": F.mse_loss,
    "l1": F.l1_loss,
    "mape": mape,
    "smape": smape,
}

MOL_LOSSES = {
    "dml": discretized_mix_logistic_loss,
    "cml": mix_gaussian_loss,
    # "clml": mix_loggaussian_loss,
    "focal_dml": partial(discretized_mix_logistic_loss, gamma=2.0),
    "focal_cml": partial(mix_gaussian_loss, gamma=2.0),
    # "focal_clml": partial(mix_loggaussian_loss, gamma=2.0),
}

CLS_LOSSES = {
    "ce": partial(F.cross_entropy, label_smoothing=0.1),
    "focal_ce": partial(focal_ce, label_smoothing=0.1),
}


def linear(x): return x


OUT_ACT = {
    "linear": linear,
    "elu": partial(F.elu, inplace=True),
    "elu10": partial(F.elu, alpha=10, inplace=True),
    "relu": partial(F.relu, inplace=True),
}


class InstanceBase(BaseModel, InstanceTrackerInterface):
    def __init__(self, option, model_type, dataset, modules):
        super().__init__(option)
        self.visual_names = ["data_visual"]

        self.loss_fns = {}
        self.has_reg_targets = dataset.has_reg_targets
        self.has_mol_targets = dataset.has_mol_targets
        self.has_cls_targets = dataset.has_cls_targets
        self.reg_targets_idx = dataset.reg_targets_idx
        self.mol_targets_idx = dataset.mol_targets_idx
        self.cls_targets_idx = dataset.cls_targets_idx

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

        if self.has_mol_targets:
            self.loss_names.append("loss_mol")

            self.get_task_weights_scale_center(
                dataset, task="mol", short_task="mol", default_norm="min-max", targets_idx=self.mol_targets_idx
            )

            self.num_mixtures = [dataset.targets[target].get("num_mixtures", 1) for target in dataset.targets
                                 if dataset.targets[target]["task"] == "mol"]
            self.num_mol_intervals = np.array(
                [dataset.targets[target].get("class_tol", .1) for target in dataset.targets
                 if dataset.targets[target]["task"] == "mol"])
            self.num_mol_intervals = np.round(self.mol_scale_targets[0] / self.num_mol_intervals)
            # make even
            self.num_mol_intervals += self.num_mol_intervals % 2

            # self.use_logspace_out = False

            loss_strs = option.get("mol_loss_fn", "dml")
            if len(loss_strs) > 0:
                loss_strs = loss_strs.split(",")
                self.loss_fns["mol"] = []
                for loss_str in loss_strs:
                    loss = MOL_LOSSES[loss_str]
                    # if loss_str == "clml": # model output directly in logspace
                    #     self.use_logspace_out = True
                    self.loss_fns["mol"].append(loss)
        else:
            self.num_mixtures = []

        if self.has_cls_targets:
            self.loss_names.append("loss_cls")
            loss_strs = option.get("cls_loss_fn", "ce")

            weights = [dataset.targets[target].get("weight", 1) for target in dataset.targets
                       if dataset.targets[target]["task"] == "classification"]
            self.register_buffer("cls_weights", torch.tensor(weights, dtype=torch.float))

            self.loss_fns["cls"] = []
            if len(loss_strs) > 0:
                loss_strs = loss_strs.split(",")
                for loss_str in loss_strs:
                    loss = CLS_LOSSES[loss_str]
                    self.loss_fns["cls"].append(loss)

        self.num_reg_classes = dataset.num_reg_classes
        self.num_mol_classes = dataset.num_mol_classes
        self.num_cls_classes = dataset.num_cls_classes

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
        reg_out = mol_out = cls_out = None
        if outputs is not None:

            if self.has_reg_targets:
                reg_out = self.reg_out_act(outputs[:, :self.num_reg_classes])
            if self.has_mol_targets:
                mol_out = outputs[:, self.num_reg_classes: self.num_reg_classes + self.num_mol_classes]
                # if self.use_logspace_out:
                #     nr_mix = mol_out.size(1) // 3
                #     mol_out[:, nr_mix:2 * nr_mix] = F.softplus(mol_out[:, nr_mix:2 * nr_mix])
            if self.has_cls_targets:
                cls_out = outputs[:, self.num_reg_classes + self.num_mol_classes:]

        return reg_out, mol_out, cls_out

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

    def compute_mol_loss(self):
        if self.has_mol_targets and len(self.loss_fns["mol"]) > 0 and self.mol_y_mask.any():
            # iterate through each mol task
            i_mixtures = 0
            self.loss_mol = 0
            for i, (num_mixtures, num_classes) in enumerate(zip(self.num_mixtures, self.num_mol_intervals)):
                mask = torch.zeros_like(self.mol_y_mask)
                # only set mask for current task
                mask[:, i: i + 1] = self.mol_y_mask[:, i: i + 1]
                if (~mask).all():
                    continue
                out_mask = torch.zeros_like(self.mol_out).bool()
                out_mask[:, i_mixtures * 3: (i_mixtures + num_mixtures) * 3] = \
                    mask[:, i: i + 1].repeat_interleave(num_mixtures * 3, 1)

                output = self.mol_out[out_mask].reshape(-1, num_mixtures * 3)
                labels = (self.mol_y[mask].reshape(-1, 1) - self.mol_center_targets[:, [i]]) / self.mol_scale_targets[:, [i]]
                labels = labels * 2 - 1  # between -1 and 1

                if self.training and self.double_batch:
                    output2 = self.mol_out2[out_mask].reshape(-1, num_mixtures * 3)

                loss_mol = 0
                for loss_fn in self.loss_fns["mol"]:
                    if self.training and self.double_batch:
                        loss_mol += (
                                (0.5 * loss_fn(output, labels, num_classes=num_classes, reduce=False)) +
                                (0.5 * loss_fn(output2, labels, num_classes=num_classes, reduce=False))
                        ).mean()
                    else:
                        loss_mol += loss_fn(output, labels, num_classes=num_classes, reduce=True)
                self.loss += self.mol_weights[i] * loss_mol
                self.loss_mol += loss_mol
                i_mixtures += num_mixtures

    def compute_cls_loss(self):
        if self.has_cls_targets and len(self.loss_fns["cls"]) > 0 and self.cls_y_mask.any():
            # iterate through each classification task
            i_classes = 0
            self.loss_cls = 0
            for i, num_classes in enumerate(self.num_cls_classes):
                mask = torch.zeros_like(self.cls_y_mask)
                # only set mask for current task
                mask[:, i: i + 1] = self.cls_y_mask[:, i: i + 1]
                if (~mask).all():
                    continue
                out_mask = torch.zeros_like(self.cls_out).bool()
                out_mask[:, i_classes: i_classes + num_classes] = mask[:, i: i + 1].repeat_interleave(num_classes, 1)

                output = self.cls_out[out_mask].reshape(-1, num_classes)
                labels = self.cls_y[mask]
                if self.training and self.double_batch:
                    output2 = self.cls_out2[out_mask].reshape(-1, num_classes)

                loss_cls = 0
                for loss_fn in self.loss_fns["cls"]:
                    if self.training and self.double_batch:
                        loss_cls += (
                                (0.5 * loss_fn(output, labels, reduction="none")) +
                                (0.5 * loss_fn(output2, labels, reduction="none"))
                        ).mean()
                    else:
                        loss_cls += loss_fn(output, labels)

                self.loss += self.cls_weights[i] * loss_cls
                self.loss_cls += loss_cls
                i_classes = i_classes + num_classes

    def get_reg_output(self):
        """ returns a tensor of size ``[N_points,N_regression_targets]`` where each value is the regression output
        of the network for a point (output of the last layer in general)
        """
        return self.reg_report_out_act(self.reg_out * self.reg_scale_targets + self.reg_center_targets)

    def get_mol_output(self, ensemble=True):
        """ returns a tensor of size ``[N_points,N_mol_targets]`` where each value is the mixture of logits output
        of the network for a point (output of the last layer in general)
        """
        mol_out = []
        i_mixtures = 0

        for i, num_mixtures in enumerate(self.num_mixtures):
            mixture = self.mol_out[:, i_mixtures * 3: (i_mixtures + num_mixtures) * 3]
            logits = mixture[:, : num_mixtures]
            means = mixture[:, num_mixtures: num_mixtures * 2]

            if ensemble:
                # ensemble mixture predictions
                softmax = logits.softmax(-1)
            else:
                # use most important mixture prediction only
                softmax = to_one_hot(logits.max(-1)[1], num_mixtures)

            mol_out.append(torch.clamp((means * softmax).sum(1), min=-1, max=1))

            i_mixtures += num_mixtures

        mol_out = torch.stack(mol_out, 1)
        return (((mol_out + 1) * self.mol_scale_targets) / 2.) + self.mol_center_targets

    def get_cls_output(self):
        """ returns a list of tensors for each classification task,
        each of size ``[N_points,...]`` where each value is the log probability output
        of the network for a point (output of the last layer in general)
        """
        cls_out = []
        cls_i = 0
        for num_cls in self.num_cls_classes:
            cls_out.append(self.cls_out[:, cls_i: cls_i + num_cls])
            cls_i += num_cls

        return cls_out

    def get_reg_input(self):
        """ returns the last regression input that was given to the model or raises error
        """
        return self.reg_y

    def get_mol_input(self):
        """ returns the last mixture of logits input that was given to the model or raises error
        """
        return self.mol_y

    def get_cls_input(self):
        """ returns the last classification input that was given to the model or raises error
        """
        return self.cls_y

    def compute_instance_loss(self):
        self.compute_reg_loss()
        self.compute_mol_loss()
        self.compute_cls_loss()

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        if not self.opt.get("override_target_stats", True):
            remove_dict_entry(state_dict, "reg_scale_targets")
            remove_dict_entry(state_dict, "reg_center_targets")
            remove_dict_entry(state_dict, "mol_scale_targets")
            remove_dict_entry(state_dict, "mol_center_targets")
            remove_dict_entry(state_dict, "reg_weights")
            remove_dict_entry(state_dict, "mol_weights")
            remove_dict_entry(state_dict, "cls_weights")

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
        self.has_mol_targets = dataset.has_mol_targets
        self.has_cls_targets = dataset.has_cls_targets

        if self.has_reg_targets or self.has_mol_targets or self.has_cls_targets:

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

            if dataset.has_mol_targets:
                self.loss_names.append("loss_mol")
                min = np.nanmean([area["train"] for area in dataset.get_min_targets().values()])
                max = np.nanmean([area["train"] for area in dataset.get_max_targets().values()])
                self.register_buffer("min_targets", torch.tensor(min.reshape(1, -1), dtype=torch.float))
                self.register_buffer("max_targets", torch.tensor(max.reshape(1, -1), dtype=torch.float))
                loss_strs = option.get("mol_loss_fn", "dml")
                if len(loss_strs) > 0:
                    loss_strs = loss_strs.split(",")
                    for loss_str in loss_strs:
                        loss = MOL_LOSSES[loss_str]
                        self.loss_fns["mol"].append(loss)

            if dataset.has_cls_targets:
                self.loss_names.append("loss_cls")
                loss_strs = option.get("cls_loss_fn", "ce")
                if len(loss_strs) > 0:
                    loss_strs = loss_strs.split(",")
                    for loss_str in loss_strs:
                        loss = CLS_LOSSES[loss_str]
                        self.loss_fns["cls"].append(loss)

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
