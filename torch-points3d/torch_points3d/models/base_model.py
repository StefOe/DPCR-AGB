import logging
import os
from collections import OrderedDict
from typing import Optional, Dict, Any, List

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


from torch_points3d.core.optimizer.adabelief import AdaBelief
from torch_points3d.core.regularizer import *
from torch_points3d.core.schedulers.bn_schedulers import instantiate_bn_scheduler
from torch_points3d.core.schedulers.lr_schedulers import instantiate_scheduler
from torch_points3d.utils.colors import colored_print, COLORS
from torch_points3d.utils.enums import SchedulerUpdateOn
from .model_interface import TrackerInterface, DatasetInterface, CheckpointInterface

log = logging.getLogger(__name__)


class BaseModel(torch.nn.Module, TrackerInterface, DatasetInterface, CheckpointInterface):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    __REQUIRED_DATA__: List[str] = []
    __REQUIRED_LABELS__: List[str] = []

    def __init__(self, opt):
        """Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super(BaseModel, self).__init__()
        self.opt = opt
        self.loss_names = []
        self.visual_names = []
        self.output = None
        self.model = None
        self._conv_type = opt.conv_type if hasattr(opt, 'conv_type') else None  # Update to OmegaConv 2.0
        self._optimizer: Optional[Optimizer] = None
        self._lr_scheduler: Optimizer[_LRScheduler] = None
        self._bn_scheduler = None
        self._spatial_ops_dict: Dict = {}
        self._num_epochs = 0
        self._num_batches = 0
        self._num_samples = -1
        self._schedulers = {}
        self._accumulated_gradient_step = 1
        self._grad_clip = -1
        self._grad_scale = None
        self._supports_mixed = False
        self._enable_mixed = False
        self._update_lr_scheduler_on = "on_epoch"
        self._update_bn_scheduler_on = "on_epoch"

    @property
    def schedulers(self):
        return self._schedulers

    @schedulers.setter
    def schedulers(self, schedulers):
        if schedulers:
            self._schedulers = schedulers
            for scheduler_name, scheduler in schedulers.items():
                setattr(self, "_{}".format(scheduler_name), scheduler)

    def _add_scheduler(self, scheduler_name, scheduler):
        setattr(self, "_{}".format(scheduler_name), scheduler)
        self._schedulers[scheduler_name] = scheduler

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def grad_scale(self):
        return self._grad_scale

    @grad_scale.setter
    def grad_scale(self, grad_scale):
        self._grad_scale = grad_scale

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs):
        self._num_epochs = num_epochs

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, num_batches):
        self._num_batches = num_batches

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples):
        self._num_samples = num_samples

    @property
    def learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def conv_type(self):
        return self._conv_type

    @conv_type.setter
    def conv_type(self, conv_type):
        self._conv_type = conv_type

    def is_mixed_precision(self):
        return self._supports_mixed and self._enable_mixed

    def set_input(self, input, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        raise NotImplementedError

    def load_state_dict_with_same_shape(self, weights, strict=False):
        model_state = self.state_dict()
        filtered_weights = {k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()}
        unmatched_weights = [k for k, v in weights.items() if k not in model_state or v.size() != model_state[k].size()]

        log.info("Loading weights:" + ", ".join(filtered_weights.keys()))
        if len(unmatched_weights) > 0:
            log.info("These weights did not match:" + ", ".join(unmatched_weights))
        self.load_state_dict(filtered_weights, strict=strict)

    def set_pretrained_weights(self):
        path_pretrained = getattr(self.opt, "path_pretrained", None)
        weight_name = getattr(self.opt, "weight_name", "latest")

        if path_pretrained is not None:
            if not os.path.exists(path_pretrained):
                raise FileNotFoundError("The path does not exist, it will not load any model")
            else:
                log.info("load pretrained weights from {}".format(path_pretrained))
                m = torch.load(path_pretrained, map_location="cpu")["models"][weight_name]
                self.load_state_dict_with_same_shape(m, strict=False)

    def get_labels(self):
        """returns a tensor of size ``[N_points]`` where each value is the label of a point"""
        return getattr(self, "labels", None)

    def get_batch(self):
        """returns a tensor of size ``[N_points]`` where each value is the batch index of a point"""
        return getattr(self, "batch_idx", None)

    def get_output(self):
        """returns a tensor of size ``[N_points,...]`` where each value is the output
        of the network for a point (output of the last layer in general)
        """
        return self.output

    def get_input(self):
        """returns the last input that was given to the model or raises error"""
        return getattr(self, "input")

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        raise NotImplementedError("You must implement your own forward")

    def _manage_optimizer_zero_grad(self):
        if self._accumulated_gradient_step == 1:
            self._optimizer.zero_grad()  # clear existing gradients
            return True
        else:
            if self._accumulated_gradient_count == self._accumulated_gradient_step:
                self._accumulated_gradient_count = 0
                return True

            if self._accumulated_gradient_count == 0:
                self._optimizer.zero_grad()  # clear existing gradients
            self._accumulated_gradient_count += 1
            return False

    def _do_scheduler_update(self, update_scheduler_on, scheduler, epoch, batch_size, num_batches):
        if hasattr(self, update_scheduler_on):
            update_scheduler_on = getattr(self, update_scheduler_on)
            if update_scheduler_on is None:
                raise Exception("The optimizer does not seems to be instantiated (instantiate_optimizers).")

            num_steps = 0
            step_size = epoch
            if update_scheduler_on == SchedulerUpdateOn.ON_EPOCH.value:
                num_steps = epoch - self._num_epochs
            elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_BATCH.value:
                num_steps = 1
                step_size = self._num_batches / num_batches
            elif update_scheduler_on == SchedulerUpdateOn.ON_NUM_SAMPLE.value:
                num_steps = batch_size

            for _ in range(num_steps):
                scheduler.step(step_size)
        else:
            raise Exception("The attributes {} should be defined within self".format(update_scheduler_on))

    def optimize_parameters(self, epoch, batch_size, num_batches):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        with torch.cuda.amp.autocast(enabled=self.is_mixed_precision()):  # enable autocasting if supported
            self(epoch=epoch)  # first call forward to calculate intermediate results

        self.loss = self._grad_scale.scale(self.loss / self._accumulated_gradient_step)  # scale losses if needed
        make_optimizer_step = self._manage_optimizer_zero_grad()  # Accumulate gradient if option is up
        self.backward()  # calculate gradients

        if make_optimizer_step:
            if self._grad_clip > 0:
                self._grad_scale.unscale_(self._optimizer)  # unscale losses to orig
                torch.nn.utils.clip_grad_value_(self.parameters(), self._grad_clip)

            self._grad_scale.step(self._optimizer)  # update parameters
            self._grad_scale.update()  # update scaling

            if self._lr_scheduler:
                self._do_scheduler_update("_update_lr_scheduler_on", self._lr_scheduler, epoch, batch_size, num_batches)

            if self._bn_scheduler:
                self._do_scheduler_update("_update_bn_scheduler_on", self._bn_scheduler, epoch, batch_size, num_batches)

        self._num_epochs = epoch
        self._num_batches += 1
        self._num_samples += batch_size

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # calculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if hasattr(self, name):
                    try:
                        errors_ret[name] = float(getattr(self, name))
                    except:
                        errors_ret[name] = None
        return errors_ret

    def get_parameter_list(self) -> List[dict]:
        return [{"params": self.parameters()}]

    def init_train_objects(self, config):
        self.init_optim(config)

        self.init_schedulers(config)

        # Accumulated gradients
        self._accumulated_gradient_step = self.get_from_opt(
            config, ["training", "optim", "accumulated_gradient"], default_value=1
        )
        if self._accumulated_gradient_step > 1:
            self._accumulated_gradient_count = 0
        # Gradient clipping
        self._grad_clip = self.get_from_opt(config, ["training", "optim", "grad_clip"], default_value=-1)

        self.init_grad_scaler(config)

    def init_optim(self, config):
        # Optimiser
        optimizer_opt = self.get_from_opt(
            config,
            ["training", "optim", "optimizer"],
            msg_err="optimizer needs to be defined within the training config",
        )
        optimizer_cls_name = optimizer_opt.get("class")
        if optimizer_cls_name == "AdaBelief":
            optimizer_cls = AdaBelief
        else:
            optimizer_cls = getattr(torch.optim, optimizer_cls_name)
        optimizer_params = {}
        if hasattr(optimizer_opt, "params"):
            optimizer_params = optimizer_opt.params
        self._optimizer = optimizer_cls(self.get_parameter_list(), **optimizer_params)

    def init_grad_scaler(self, config):
        # Gradient Scaling
        self._enable_mixed = self.get_from_opt(config, ["training", "enable_mixed"], default_value=False)
        self._enable_mixed = bool(self._enable_mixed)
        if self._enable_mixed and not self._supports_mixed:
            self._enable_mixed = False
            log.warning("Mixed precision is not supported on this model, using default precision...")
        elif self.is_mixed_precision():
            log.info("Model will use mixed precision")
        self._grad_scale = torch.cuda.amp.GradScaler(enabled=self.is_mixed_precision())

    def init_schedulers(self, config):
        # LR Scheduler
        scheduler_opt = self.get_from_opt(config, ["training", "optim", "lr_scheduler"])
        if scheduler_opt:
            update_lr_scheduler_on = config.get('update_lr_scheduler_on')  # Update to OmegaConf 2.0
            if update_lr_scheduler_on:
                self._update_lr_scheduler_on = update_lr_scheduler_on
            scheduler_opt.update_scheduler_on = self._update_lr_scheduler_on
            lr_scheduler = instantiate_scheduler(self._optimizer, scheduler_opt)
            self._add_scheduler("lr_scheduler", lr_scheduler)
        # BN Scheduler
        bn_scheduler_opt = self.get_from_opt(config, ["training", "optim", "bn_scheduler"])
        if bn_scheduler_opt:
            update_bn_scheduler_on = config.get('update_bn_scheduler_on')  # update to OmegaConf 2.0
            if update_bn_scheduler_on:
                self._update_bn_scheduler_on = update_bn_scheduler_on
            bn_scheduler_opt.update_scheduler_on = self._update_bn_scheduler_on
            bn_scheduler = instantiate_bn_scheduler(self, bn_scheduler_opt)
            self._add_scheduler("bn_scheduler", bn_scheduler)

    def get_regularization_loss(self, regularizer_type="L2", **kwargs):
        loss = 0
        regularizer_cls = RegularizerTypes[regularizer_type.upper()].value
        regularizer = regularizer_cls(self, **kwargs)
        return regularizer.regularized_all_param(loss)

    def get_spatial_ops(self):
        return self._spatial_ops_dict

    def enable_dropout_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if "Dropout" in m.__class__.__name__:
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)

    def enable_bn_in_eval(self):
        def search_from_key(modules):
            for _, m in modules.items():
                if "BatchNorm" in m.__class__.__name__:
                    m.train()
                search_from_key(m._modules)

        search_from_key(self._modules)

    def get_from_opt(self, opt, keys=[], default_value=None, msg_err=None, silent=True):
        if len(keys) == 0:
            raise Exception("Keys should not be empty")
        value_out = default_value

        def search_with_keys(args, keys, value_out):
            if len(keys) == 0:
                value_out = args
                return value_out
            value = args[keys[0]]
            return search_with_keys(value, keys[1:], value_out)

        try:
            value_out = search_with_keys(opt, keys, value_out)
        except Exception as e:
            if msg_err:
                raise Exception(str(msg_err))
            else:
                if not silent:
                    log.exception(e)
            value_out = default_value
        return value_out

    def get_current_visuals(self):
        """Return an OrderedDict containing associated tensors within visual_names"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def log_optimizers(self):
        colored_print(COLORS.Green, "Optimizer: {}".format(self._optimizer))
        colored_print(COLORS.Green, "Learning Rate Scheduler: {}".format(self._lr_scheduler))
        colored_print(COLORS.Green, "BatchNorm Scheduler: {}".format(self._bn_scheduler))
        colored_print(COLORS.Green, "Accumulated gradients: {}".format(self._accumulated_gradient_step))

    def to(self, *args, **kwargs):
        super().to(*args, *kwargs)
        if self.optimizer:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(*args, **kwargs)
        return self

    def verify_data(self, data, forward_only=False):
        """Goes through the __REQUIRED_DATA__ and __REQUIRED_LABELS__ attribute of the model
        and verifies that the passed data object contains all required members.
        If something is missing it raises a KeyError exception.
        """
        missing_keys = []
        required_attributes = self.__REQUIRED_DATA__
        if not forward_only:
            required_attributes += self.__REQUIRED_LABELS__
        for attr in required_attributes:
            if not hasattr(data, attr) or data[attr] is None:
                missing_keys.append(attr)
        if len(missing_keys):
            raise KeyError(
                "Missing attributes in your data object: {}. The model will fail to forward.".format(missing_keys)
            )

    def print_transforms(self):
        message = ""
        for attr in self.__dict__:
            if "transform" in attr:
                message += "{}{} {}= {}\n".format(COLORS.IPurple, attr, COLORS.END_NO_TOKEN, getattr(self, attr))
        print(message)
