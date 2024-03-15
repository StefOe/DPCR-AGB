import logging
from collections import OrderedDict
from typing import Dict, Any

import numpy as np
import torch
import wandb
from torchnet.meter import MSEMeter

from torch_points3d.metrics.base_tracker import BaseTracker
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.meters.maemeter import MAEMeter
from torch_points3d.metrics.meters.r2meter import R2Meter
from torch_points3d.models import model_interface


class InstanceTracker(BaseTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 log_train_metrics: bool = True):
        """ This is a generic tracker for instance prediction tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch
        Arguments:
            dataset  -- dataset to track (used for the number of classes)
        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(InstanceTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self.has_reg_targets = dataset.has_reg_targets
        self.reg_targets_idx = dataset.reg_targets_idx
        self.reg_targets = dataset.reg_targets

        self.area_names = dataset.areas.keys()
        self.area_name_map = OrderedDict({area_name: i for i, area_name in enumerate(self.area_names)})

        self.n_targets = dataset.num_classes

        # for r2 score
        self.target_means = dataset.get_mean_targets()
        self.log_train_metrics = log_train_metrics

        self.reset(stage)
        # Those map subsentences to their optimization functions
        self._metric_goals = {
            "loss": "minimize",
        }
        self._metric_func = {
            "loss": min,
        }
        if self.has_reg_targets:
            self._metric_goals.update({
                "_rmse": "minimize",
                "_mae": "minimize",
                "_r2": "maximize",
            })
            self._metric_func.update({
                "_rmse": min,
                # "mae": min,
                # "r2": max,
            })
        if self.has_reg_targets:
            self._metric_func.update({"loss_reg": min})

        if wandb_log:
            self.wandb_metrics = []

    def reset(self, stage="train"):
        super().reset(stage=stage)
        if (stage == "train" and self.log_train_metrics) or stage != "train":
            area_names = [area_name for area_name in self.area_names
                          if self.target_means[area_name].get(stage, None) is not None]
            area_names.append("total")
            if self.has_reg_targets:
                targets = self.reg_targets
                targets_idx = self.reg_targets_idx
                self._rmse = {area_name: {} for area_name in area_names}
                self._mae = {area_name: {} for area_name in area_names}
                self._r2 = {area_name: {} for area_name in area_names}
                for i, target_name in enumerate(targets):
                    for area_name in area_names:
                        if np.isnan(self.target_means[area_name][stage][targets_idx][i]).all():
                            continue
                        self._rmse[area_name][target_name] = MSEMeter(root=True)
                        self._mae[area_name][target_name] = MAEMeter()
                        self._r2[area_name][target_name] = R2Meter(self.target_means[area_name][stage][targets_idx][i])

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    def track(self, model: model_interface.InstanceTrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model)

        if (self._stage == "train" and self.log_train_metrics) or self._stage != "train":
            areas = model.data_visual["area_name"]
            areas = torch.tensor([self.area_name_map[an] for an in areas])

            # regression
            if self.has_reg_targets:
                outputs = model.get_reg_output()
                targets = model.get_reg_input()

                track_stats = self.track_numerical_stats
                target_names = self.reg_targets

                self.track_iterate_areas_targets(areas, outputs, target_names, targets, track_stats)

    def track_iterate_areas_targets(self, areas, outputs, target_names, targets, track_stats):
        # ignore nan values
        targets_nan = torch.isnan(targets) if targets.dtype == torch.float else targets == -1
        no_nans = ~targets_nan  # ~(outputs_nan | targets_nan)
        if no_nans.any():
            for i, target_name in enumerate(target_names):
                no_nan = no_nans[:, i]
                # skip if no real values are present
                if not no_nan.any():
                    continue
                out = outputs[:, i][no_nan]
                target = targets[:, i][no_nan]
                area = areas[no_nan.cpu()]

                for area_name in self.area_names:
                    area_idx = area == self.area_name_map[area_name]
                    if area_idx.any():
                        track_stats(area_idx, area_name, out, target, target_name)
                track_stats(torch.ones_like(area_idx), "total", out, target, target_name)

    def track_numerical_stats(self, area_idx, area_name, out, target, target_name):
        self._rmse[area_name][target_name].add(out[area_idx], target[area_idx])
        self._mae[area_name][target_name].add(out[area_idx], target[area_idx])
        self._r2[area_name][target_name].add(out[area_idx], target[area_idx])

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionary of all metrics and losses being tracked
        """
        metrics = super().get_loss()
        if (self._stage == "train" and self.log_train_metrics) or self._stage != "train":
            area_names = list(self.area_names)
            area_names.append("total")
            for area_name in area_names:
                if self.has_reg_targets:
                    if self._r2.get(area_name, None) is not None:
                        for target_name in self.reg_targets:
                            if self._r2[area_name].get(target_name, None) is None:
                                continue
                            metrics[f"{self._stage}_{area_name}_{target_name}_rmse"] = \
                                self._rmse[area_name][target_name].value()
                            metrics[f"{self._stage}_{area_name}_{target_name}_mae"] = \
                                self._mae[area_name][target_name].value()
                            metrics[f"{self._stage}_{area_name}_{target_name}_r2"] = \
                                self._r2[area_name][target_name].value()

        if self._wandb:
            # add metric to wandb if not there already
            new_metrics = [metric for metric in metrics if metric not in self.wandb_metrics]
            for metric in new_metrics:
                m_func = [m for m in self._metric_goals if m in metric]
                if len(m_func) == 0:
                    m_func = goal = None
                else:
                    try:
                        m_func, goal = self._metric_goals[m_func[0]][:3], self._metric_goals[m_func[0]]
                    except Exception as e:
                        logging.warning(f"{str(e)}\n Something went wrong during wandb metric collection")
                wandb.define_metric(metric, step_metric="epoch", summary=m_func, goal=goal)
                self.wandb_metrics.append(metric)

        return metrics

    @property
    def metric_func(self):
        return self._metric_func
