import math
import sys
import warnings
from typing import List

from torch.optim import lr_scheduler, Optimizer
from omegaconf import OmegaConf
import logging
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from torch_points3d.utils.enums import SchedulerUpdateOn

log = logging.getLogger(__name__)

_custom_lr_scheduler = sys.modules[__name__]


def collect_params(params, update_scheduler_on):
    """
    This function enable to handle if params contains on_epoch and on_iter or not.
    """
    on_epoch_params = params.get("on_epoch")
    on_batch_params = params.get("on_num_batch")
    on_sample_params = params.get("on_num_sample")

    def check_params(params):
        if params is not None:
            return params
        else:
            raise Exception(
                "The lr_scheduler doesn't have policy {}. Options: {}".format(update_scheduler_on, SchedulerUpdateOn)
            )

    if on_epoch_params or on_batch_params or on_sample_params:
        if update_scheduler_on == SchedulerUpdateOn.ON_EPOCH.value:
            return check_params(on_epoch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_BATCH.value:
            return check_params(on_batch_params)
        elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_SAMPLE.value:
            return check_params(on_sample_params)
        else:
            raise Exception(
                "The provided update_scheduler_on {} isn't within {}".format(update_scheduler_on, SchedulerUpdateOn)
            )
    else:
        return params


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.00003,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        from https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                    1
                    + math.cos(
                math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        lambda_func = lambda s: (1 - s / (max_iter + 1)) ** power
        composite_func = lambda s: lambda_func(max_iter) if s > max_iter else lambda_func(s)
        super(PolyLR, self).__init__(optimizer, lambda s: composite_func(s), last_step)


class SquaredLR(LambdaStepLR):
    """ Used for SGD Lars"""

    def __init__(self, optimizer, max_iter, last_step=-1):
        super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1)) ** 2, last_step)


class ExpLR(LambdaStepLR):
    def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
        # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
        # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
        # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
        super(ExpLR, self).__init__(optimizer, lambda s: gamma ** (s / step_size), last_step)


def repr(self, scheduler_params={}):
    return "{}({})".format(self.__class__.__name__, scheduler_params)


class LRScheduler:
    def __init__(self, scheduler, scheduler_params, update_scheduler_on):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._update_scheduler_on = update_scheduler_on

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return self._scheduler._scheduler_opt

    def __repr__(self):
        return "{}({}, update_scheduler_on={})".format(
            self._scheduler.__class__.__name__, self._scheduler_params, self._update_scheduler_on
        )

    def step(self, *args, **kwargs):
        self._scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


def instantiate_scheduler(optimizer, scheduler_opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        scheduler_opt (option class) -- dict containing all the params to build the scheduler　
                              opt.lr_policy is the name of learning rate policy: lambda_rule | step | plateau | cosine
                              opt.params contains the scheduler_params to construct the scheduler
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    update_scheduler_on = scheduler_opt.update_scheduler_on
    scheduler_cls_name = getattr(scheduler_opt, "class")
    scheduler_params = collect_params(scheduler_opt.params, update_scheduler_on)

    try:
        scheduler_cls = getattr(lr_scheduler, scheduler_cls_name)
    except:
        scheduler_cls = getattr(_custom_lr_scheduler, scheduler_cls_name)
        log.info("Created custom lr scheduler")

    if scheduler_cls_name.lower() == "ReduceLROnPlateau".lower():
        raise NotImplementedError("This scheduler is not fully supported yet")

    scheduler = scheduler_cls(optimizer, **scheduler_params)
    # used to re_create the scheduler
    # instantiate vars
    for key in scheduler_params.keys():
        scheduler_params[key] = getattr(scheduler, key)

    setattr(scheduler, "_scheduler_opt", OmegaConf.to_container(scheduler_opt))
    return LRScheduler(scheduler, scheduler_params, update_scheduler_on)
