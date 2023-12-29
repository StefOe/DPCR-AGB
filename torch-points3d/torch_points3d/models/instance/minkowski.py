import logging
from typing import List

import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from torch_points3d.models.instance.base import InstanceBase
from torch_points3d.models.instance.semi_supervised_helper import gather, invariance_loss, variance_loss, \
    covariance_loss, barlow_loss
from torch_points3d.modules.MinkowskiEngine import initialize_minkowski_unet

log = logging.getLogger(__name__)


class SeparateLinear(torch.nn.Module):

    def __init__(self, in_channel, num_reg_classes, num_mixtures, num_cls_classes):
        super(SeparateLinear, self).__init__()
        self.linears = []
        if num_reg_classes > 0:
            self.linears += [torch.nn.Linear(in_channel, 1, bias=True) for i in range(num_reg_classes)]
        if len(num_mixtures) > 0:
            self.linears += [
                torch.nn.Linear(in_channel, num_mixtures * 3, bias=True) for i, num_mixtures in enumerate(num_mixtures)
            ]
        if len(num_cls_classes) > 0:
            self.linears += [
                torch.nn.Linear(in_channel, num_classes) for num_classes in num_cls_classes
            ]

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
        self.model.final = SeparateLinear(in_channel, self.num_reg_classes, self.num_mixtures, self.num_cls_classes)

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


class MinkowskiVAEm2(InstanceBase):
    # similar to Kingma et al. https://proceedings.neurips.cc/paper/2014/file/d523773c6b194f37b938d340d5d02232-Paper.pdf

    def __init__(self, option, model_type, dataset, modules):
        super(MinkowskiVAEm2, self).__init__(option, model_type, dataset, modules)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension, dataset.num_classes, activation=option.activation,
            z_channels=option.z_channels, dropout=option.dropout, backbone=option.backbone,
            resolution=int(1 / dataset.dataset_opt.first_subsampling), first_stride=option.first_stride,
            global_pool=option.global_pool, **option.get("extra_options", {})
        )
        self.loss_names.extend(
            ["loss_vae", "loss_BCE", "loss_KLD", "loss_rec", "loss_r_entropy", "loss_r_cross_entropy"]
        )
        self.KLD_beta = option.KLD_beta
        self.reconstruction_beta = option.reconstruction_beta
        self.regression_beta = option.regression_beta
        self.num_reg_classes = dataset.num_reg_classes
        self.num_mol_classes = dataset.num_mol_classes
        self.num_cls_classes = dataset.num_cls_classes

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()
        coords = torch.cat([data.batch.unsqueeze(-1).int(), data.coords.int()], -1)
        self.data_visual = data
        self.input = ME.SparseTensor(features=data.x, coordinates=coords, device=device)
        self.input_target = self.input.coordinate_map_key
        self.labels_mask = data.y_mask.to(device).view(-1, self.model.out_channels)
        self.labels = data.y.to(device).view(-1, self.model.out_channels)

    def compute_loss(self):
        # VAE loss
        # loss to check if correct pruning was applied

        self.loss_BCE = 0
        for out_cl, target in zip(self.out_cls, self.rec_targets):
            curr_loss = F.binary_cross_entropy_with_logits(out_cl.F.squeeze(), target.type(out_cl.F.dtype))
            self.loss_BCE += curr_loss / len(self.out_cls)

        self.loss_KLD = self.KLD_beta * 0.5 * torch.mean(
            torch.mean(self.z_logvar.F.exp() + self.z_mean.F.pow(2) - 1 - self.z_logvar.F, 1)
        )
        # feature reconstruction error (removes last dim as it is assumed to be 1)
        rec = self.reconstruction
        rec.F[:, -1] = 1

        loss_rec = (rec - self.input).F
        loss_rec = loss_rec[loss_rec[:, -1] == 0][:, :-1]
        use_l1 = loss_rec > 1.0
        self.loss_rec = (torch.pow(loss_rec, 2) * ~use_l1 + torch.abs(loss_rec) * use_l1).mean()

        self.loss_vae = self.loss_KLD + self.reconstruction_beta * (self.loss_BCE + self.loss_rec)

        self.loss_regr = 0
        self.loss_r_cross_entropy = 0
        self.loss_r_entropy = 0
        # only calculate loss if labels are set
        if self.labels is not None:
            if self.labels_mask.any():
                # scaling by std to have equal grads
                cond = (self.cond.F / self.reg_scale_targets)[self.labels_mask]
                labels = (self.labels / self.reg_scale_targets)[self.labels_mask]
                for loss_fn in self.loss_fns:
                    self.loss_regr += loss_fn(cond, labels) / len(self.loss_fns)  # scaling by std to have equal grads

        self.loss = self.regression_beta * (self.loss_regr + self.loss_r_entropy + self.loss_r_cross_entropy) \
                    + self.loss_vae

        return self.loss

    def forward(self, *args, **kwargs):
        (self.out_cls, self.rec_targets, self.reconstruction,
         self.zs, self.z_mean, self.z_logvar,
         self.cond_norm, self.cond) = self.model(
            self.input, self.input_target, self.labels, self.labels_mask
        )
        self.output = self.cond.F
        self.compute_loss()

        self.data_visual.pred = self.output


class MinkowskiVAE(MinkowskiVAEm2):
    # similar to VAE for regression paper https://arxiv.org/abs/1904.05948

    def compute_loss(self):
        # VAE loss
        # loss to check if correct pruning was applied

        self.loss_BCE = 0
        for out_cl, target in zip(self.out_cls, self.rec_targets):
            curr_loss = F.binary_cross_entropy_with_logits(out_cl.F.squeeze(), target.type(out_cl.F.dtype))
            self.loss_BCE += curr_loss / len(self.out_cls)

        self.loss_KLD = self.KLD_beta * 0.5 * torch.mean(
            torch.mean(self.z_logvar.F.exp() + self.z_mean.F.pow(2) - 1 - self.z_logvar.F, 1)
        )
        # feature reconstruction error (removes last dim as it is assumed to be 1)
        rec = self.reconstruction
        rec.F[:, -1] = 1

        loss_rec = (rec - self.input).F  # TODO maybe just include intersected points
        loss_rec = loss_rec[loss_rec[:, -1] == 0][:, :-1]
        use_l1 = loss_rec > 1.0
        self.loss_rec = (torch.pow(loss_rec, 2) * ~use_l1 + torch.abs(loss_rec) * use_l1).mean()

        self.loss_vae = self.loss_KLD + self.reconstruction_beta * (self.loss_BCE + self.loss_rec)

        self.loss_regr = 0
        self.loss_r_cross_entropy = 0
        self.loss_r_entropy = 0
        # only calculate loss if labels are set
        if self.labels is not None:
            shape = self.r_mean.F.shape
            if self.labels_mask.any():
                r_mean = (self.r_mean.F / self.reg_scale_targets)[self.labels_mask]
                r_logvar = (self.r_logvar.F / self.reg_scale_targets.pow(2).log())[self.labels_mask]
                labels = (self.labels / self.reg_scale_targets)[self.labels_mask]
                for loss_fn in self.loss_fns:
                    self.loss_regr += ((loss_fn(r_mean, labels, reduction="none")
                                        / (r_logvar.detach().exp() ** 0.5)).mean()  # scaling by std to have equal grads
                                       / len(self.loss_fns))

                # cross entropy
                self.loss_r_cross_entropy += 0.5 * (
                        ((F.mse_loss(r_mean, labels - 1e-6 / shape[0], reduction="none"))
                         / (torch.exp(r_logvar))) + (r_logvar)).mean()

            # use entropy if nans are present (assumes univariate Gaussians)
            if not self.labels_mask.all():
                # r_mean = self.r_mean.F[~self.labels_mask.view(shape)]
                r_logvar = self.r_logvar.F[~self.labels_mask]
                # removing static vars
                # self.loss_r_entropy += 0.5 * (r_logvar).mean()
                '''
                intuition behind including the r_mean part: keeping the prediction constant but since the 
                factor is small, the decoder should still be able to change it (aka it is a regularization term)
                '''
                self.loss_r_entropy += 0.5 * (r_logvar).mean()
                # self.loss_r_entropy += 0.5 * (((F.smooth_l1_loss(r_mean, r_mean.detach() + 1e-6, reduction="none"))
                #                                / (torch.exp(r_logvar))) + r_logvar).mean()
                # http://gregorygundersen.com/blog/2020/09/01/gaussian-entropy
                # self.loss_r_entropy += 0.5 * ( r_logvar + np.log(2) + np.log(np.pi)) + 0.5
                # this would be wikipedia
                # self.loss_entropy += r_logvar + np.log(np.sqrt(2 * np.pi * np.e))

        self.loss = self.regression_beta * (self.loss_regr + self.loss_r_entropy + self.loss_r_cross_entropy) \
                    + self.loss_vae

        return self.loss

    def forward(self, *args, **kwargs):
        (self.out_cls, self.rec_targets, self.reconstruction,
         self.zs, self.z_mean, self.z_logvar,
         self.rs, self.r_mean, self.r_logvar
         ) = self.model(self.input, self.input_target)
        self.output = self.r_mean.F
        self.compute_loss()

        self.data_visual.pred = self.output


class MinkowskiBarlowTwins(InstanceBase):
    def __init__(self, option, model_type, dataset, modules):
        super(MinkowskiBarlowTwins, self).__init__(option, model_type, dataset, modules)
        model_version = option.get("model_version", "standard")
        self.reset_output = option.get("reset_output", True)
        self.model = initialize_minkowski_unet(
            option.model_name, dataset.feature_dimension,
            {
                "num_reg_classes": self.num_reg_classes,
                "num_mixtures": self.num_mixtures,
                "num_cls_classes": self.num_cls_classes
            },
            activation=option.activation,
            first_stride=option.first_stride, dropout=option.dropout, global_pool=option.global_pool,
            mode=option.mode, model_version=model_version, proj_activation=option.proj_activation,
            proj_layers=option.proj_layers, proj_last_norm=option.proj_last_norm, backbone=option.backbone,
            detach_classifier=option.mode != "finetune" and model_version == "standard",
            **option.get("extra_options", {})
        )

        self.mode = option.mode
        if self.mode not in ["finetune", "freeze"]:
            self.loss_names.extend(
                ["loss_self_supervised"]
            )
        self.scale_loss = option.scale_loss
        self.backbone_lr = option.backbone_lr
        self._supports_mixed = True

    def get_parameter_list(self) -> List[dict]:
        params_list = []
        classifier_parameters, model_parameters = [], []
        for name, param in self.model.named_parameters():
            if "encoder.final.classifier.linears" in name:
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
            for m in self.model.encoder.final.classifier.linears:
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def set_input(self, data, device):
        self.batch_idx = data.batch.squeeze()

        if self.training and self.double_batch:
            # augment data twice
            data = data.to_data_list()
            x1 = Batch.from_data_list(data[::2])
            x2 = Batch.from_data_list(data[1::2])
            coords2 = torch.cat([x2.batch.unsqueeze(-1).int(), x2.coords.int()], -1)
            self.input2 = ME.SparseTensor(features=x2.x, coordinates=coords2, device=device)
        else:
            x1 = data
            self.input2 = None

        bs = len(x1)
        coords = torch.cat([x1.batch.unsqueeze(-1).int(), x1.coords.int()], -1)
        self.data_visual = x1

        self.input = ME.SparseTensor(features=x1.x, coordinates=coords, device=device)

        if len(self.loss_fns) > 0:
            if self.has_reg_targets and x1.y_reg is not None:
                self.reg_y_mask = x1.y_reg_mask.to(device).view(bs, -1)
                self.reg_y = x1.y_reg.to(device).view(bs, -1)
            if self.has_mol_targets and x1.y_mol is not None:
                self.mol_y_mask = x1.y_mol_mask.to(device).view(bs, -1)
                self.mol_y = x1.y_mol.to(device).view(bs, -1)
            if self.has_cls_targets and x1.y_cls is not None:
                self.cls_y_mask = x1.y_cls_mask.to(device).view(bs, -1)
                self.cls_y = x1.y_cls.to(device).view(bs, -1)

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

    def forward(self, *args, **kwargs):
        self.set_mode()
        self.output, self.output2, self.z1, self.z2 = self.model(self.input, self.input2)
        self.reg_out, self.mol_out, self.cls_out = self.convert_outputs(self.output)
        self.reg_out2, self.mol_out2, self.cls_out2 = self.convert_outputs(self.output2)

        self.compute_loss()
        self.data_visual.pred = self.output

    def set_mode(self):
        if self.training:
            if self.mode == "freeze":
                self.model.requires_grad_(False)
                self.model.encoder.final.classifier.requires_grad_(True)
                self.model.encoder.eval()
                self.model.encoder.final.classifier.train()
                self.enable_dropout_in_eval()


class MinkowskiVICReg(MinkowskiBarlowTwins):

    def __init__(self, option, model_type, dataset, modules):
        super(MinkowskiVICReg, self).__init__(option, model_type, dataset, modules)

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

    # def calc_VICReg_loss(self, z1, z2, G=None):
    #     # following https://arxiv.org/pdf/2205.11508.pdf
    #     N, D = z1.size()
    #     V = 2
    #     if G is None:
    #         G = torch.zeros(V*N, V*N) # X′ ∈ R N ′×D′, V is the number of views
    #         i = torch.arange(0, N * V).repeat_interleave(V - 1) # row indices
    #         j= (i + torch.arange(1, V).repeat(N * V) * N).remainder(N * V)  # column indices
    #         G[i, j] = 1  # unweighted graph connecting the rows of View_1(X′ ), . . . , View_V (X′)
    #
    #     C = torch.cov(z.t())
    #     eps = 1e-4
    #     self.loss_variance += D - torch.diag(C).clamp(eps).sqrt().sum()
    #     i, j = G.nonzero(as_tuple=True)
    #     self.loss_invariance += (z[i] - z[j]).square().sum().inner(G[i, j]) / N
    #     self.loss_covariance += 2 * torch.triu(C, diagonal=1).square().sum()
