import math
import os
import random
import re
from functools import partial
from glob import glob
from itertools import chain
from pathlib import Path as PPath
from typing import List

import numba
import numpy as np
import torch
from dbscan1d.core import DBSCAN1D
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
from omegaconf import OmegaConf
from sklearn.cluster import OPTICS
from sklearn.neighbors import KDTree, KernelDensity
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import FixedPoints as FP
from tqdm.auto import tqdm as tq

from torch_points3d.datasets.multiscale_data import MultiScaleData
from torch_points3d.utils.transform_utils import SamplingStrategy
from .features import Random3AxisRotation
from .grid_transform import GridSampling3D, shuffle_data
from ...utils import is_iterable

KDTREE_KEY = "kd_tree"


class RemoveAttributes(object):
    """This transform allows to remove unnecessary attributes from data for optimization purposes

    Parameters
    ----------
    attr_names: list
        Remove the attributes from data using the provided `attr_name` within attr_names
    strict: bool=False
        Wether True, it will raise an execption if the provided attr_name isn t within data keys.
    """

    def __init__(self, attr_names=[], strict=False):
        self._attr_names = attr_names
        self._strict = strict

    def __call__(self, data):
        keys = set(data.keys)
        for attr_name in self._attr_names:
            if attr_name not in keys and self._strict:
                raise Exception("attr_name: {} isn t within keys: {}".format(attr_name, keys))
        for attr_name in self._attr_names:
            delattr(data, attr_name)
        return data

    def __repr__(self):
        return "{}(attr_names={}, strict={})".format(self.__class__.__name__, self._attr_names, self._strict)


class PointCloudFusion(object):
    """This transform is responsible to perform a point cloud fusion from a list of data

    - If a list of data is provided -> Create one Batch object with all data
    - If a list of list of data is provided -> Create a list of fused point cloud
    """

    def _process(self, data_list):
        if len(data_list) == 0:
            return Data()
        data = Batch.from_data_list(data_list)
        delattr(data, "batch")
        delattr(data, "ptr")
        return data

    def __call__(self, data_list: List[Data]):
        if len(data_list) == 0:
            raise Exception("A list of data should be provided")
        elif len(data_list) == 1:
            return data_list[0]
        else:
            if isinstance(data_list[0], list):
                data = [self._process(d) for d in data_list]
            else:
                data = self._process(data_list)
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GridSphereSampling(object):
    """Fits the point cloud to a grid and for each point in this grid,
    create a sphere with a radius r

    Parameters
    ----------
    radius: float
        Radius of the sphere to be sampled.
    grid_size: float, optional
        Grid_size to be used with GridSampling3D to select spheres center. If None, radius will be used
    delattr_kd_tree: bool, optional
        If True, KDTREE_KEY should be deleted as an attribute if it exists
    center: bool, optional
        If True, a centre transform is apply on each sphere.
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, grid_size=None, delattr_kd_tree=True, center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        grid_size = eval(grid_size) if isinstance(grid_size, str) else float(grid_size)
        self._grid_sampling = GridSampling3D(size=grid_size if grid_size else self._radius)
        self._delattr_kd_tree = delattr_kd_tree
        self._center = center

    def _process(self, data):
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        # The kdtree has bee attached to data for optimization reason.
        # However, it won't be used for down the transform pipeline and should be removed before any collate func call.
        if hasattr(data, self.KDTREE_KEY) and self._delattr_kd_tree:
            delattr(data, self.KDTREE_KEY)

        # apply grid sampling
        grid_data = self._grid_sampling(data.clone())

        datas = []
        for grid_center in np.asarray(grid_data.pos):
            pts = np.asarray(grid_center)[np.newaxis]

            # Find closest point within the original data
            ind = torch.LongTensor(tree.query(pts, k=1)[1][0])
            grid_label = data.y[ind]

            # Find neighbours within the original data
            ind = torch.LongTensor(tree.query_radius(pts, r=self._radius)[0])
            sampler = SphereSampling(self._radius, grid_center, align_origin=self._center)
            new_data = sampler(data)
            new_data.center_label = grid_label

            datas.append(new_data)
        return datas

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={})".format(self.__class__.__name__, self._radius, self._center)


class GridCylinderSampling(object):
    """Fits the point cloud to a grid and for each point in this grid,
    create a cylinder with a radius r

    Parameters
    ----------
    radius: float
        Radius of the cylinder to be sampled.
    grid_size: float, optional
        Grid_size to be used with GridSampling3D to select cylinders center. If None, radius will be used
    delattr_kd_tree: bool, optional
        If True, KDTREE_KEY should be deleted as an attribute if it exists
    center: bool, optional
        If True, a centre transform is apply on each cylinder.
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, grid_size=None, delattr_kd_tree=True, center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        grid_size = eval(grid_size) if isinstance(grid_size, str) else float(grid_size)
        self._grid_sampling = GridSampling3D(size=grid_size if grid_size else self._radius)
        self._delattr_kd_tree = delattr_kd_tree
        self._center = center

    def _process(self, data):
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=50)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        # The kdtree has bee attached to data for optimization reason.
        # However, it won't be used for down the transform pipeline and should be removed before any collate func call.
        if hasattr(data, self.KDTREE_KEY) and self._delattr_kd_tree:
            delattr(data, self.KDTREE_KEY)

        # apply grid sampling
        grid_data = self._grid_sampling(data.clone())

        datas = []
        for grid_center in np.unique(grid_data.pos[:, :-1], axis=0):
            pts = np.asarray(grid_center)[np.newaxis]

            # Find closest point within the original data
            ind = torch.LongTensor(tree.query(pts, k=1)[1][0])
            grid_label = data.y[ind]

            # Find neighbours within the original data
            ind = torch.LongTensor(tree.query_radius(pts, r=self._radius)[0])
            sampler = CylinderSampling(self._radius, grid_center, align_origin=self._center)
            new_data = sampler(data)
            new_data.center_label = grid_label

            datas.append(new_data)
        return datas

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={})".format(self.__class__.__name__, self._radius, self._center)


class ComputeKDTree(object):
    """Calculate the KDTree and saves it within data

    Parameters
    -----------
    leaf_size:int
        Size of the leaf node.
    """

    def __init__(self, leaf_size):
        self._leaf_size = leaf_size

    def _process(self, data):
        data.kd_tree = KDTree(np.asarray(data.pos), leaf_size=self._leaf_size)
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(leaf_size={})".format(self.__class__.__name__, self._leaf_size)


class RandomSphere(object):
    """Select points within a sphere of a given radius. The centre is chosen randomly within the point cloud.

    Parameters
    ----------
    radius: float
        Radius of the sphere to be sampled.
    strategy: str
        choose between `random` and `freq_class_based`. The `freq_class_based` \
        favors points with low frequency class. This can be used to balance unbalanced datasets
    center: bool
        if True then the sphere will be moved to the origin
    """

    def __init__(self, radius, strategy="random", class_weight_method="sqrt", center=True):
        self._radius = eval(radius) if isinstance(radius, str) else float(radius)
        self._sampling_strategy = SamplingStrategy(strategy=strategy, class_weight_method=class_weight_method)
        self._center = center

    def _process(self, data):
        # apply sampling strategy
        random_center = self._sampling_strategy(data)
        random_center = np.asarray(data.pos[random_center])[np.newaxis]
        sphere_sampling = SphereSampling(self._radius, random_center, align_origin=self._center)
        return sphere_sampling(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(radius={}, center={}, sampling_strategy={})".format(
            self.__class__.__name__, self._radius, self._center, self._sampling_strategy
        )


class SphereSampling:
    """ Samples points within a sphere

    Parameters
    ----------
    radius : float
        Radius of the sphere
    sphere_centre : torch.Tensor or np.array
        Centre of the sphere (1D array that contains (x,y,z))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, sphere_centre, align_origin=True):
        self._radius = radius
        self._centre = np.asarray(sphere_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])
        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the sphere.
                    item -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )


class CylinderSampling:
    """ Samples points within a cylinder

    Parameters
    ----------
    radius : float
        Radius of the cylinder
    cylinder_centre : torch.Tensor or np.array
        Centre of the cylinder (1D array that contains (x,y,z) or (x,y))
    align_origin : bool, optional
        move resulting point cloud to origin
    """

    KDTREE_KEY = KDTREE_KEY

    def __init__(self, radius, cylinder_centre, align_origin=True):
        self._radius = radius
        if cylinder_centre.shape[0] == 3:
            cylinder_centre = cylinder_centre[:-1]
        self._centre = np.asarray(cylinder_centre)
        if len(self._centre.shape) == 1:
            self._centre = np.expand_dims(self._centre, 0)
        self._align_origin = align_origin

    def __call__(self, data):
        num_points = data.pos.shape[0]
        if not hasattr(data, self.KDTREE_KEY):
            tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=50)
            setattr(data, self.KDTREE_KEY, tree)
        else:
            tree = getattr(data, self.KDTREE_KEY)

        t_center = torch.FloatTensor(self._centre)
        ind = torch.LongTensor(tree.query_radius(self._centre, r=self._radius)[0])

        new_data = Data()
        for key in set(data.keys):
            if key == self.KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[ind]
                if self._align_origin and key == "pos":  # Center the cylinder.
                    item[:, :-1] -= t_center
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data

    def __repr__(self):
        return "{}(radius={}, center={}, align_origin={})".format(
            self.__class__.__name__, self._radius, self._centre, self._align_origin
        )


class Select:
    """ Selects given points from a data object

    Parameters
    ----------
    indices : torch.Tensor
        indeices of the points to keep. Can also be a boolean mask
    """

    def __init__(self, indices=None):
        self._indices = indices

    def __call__(self, data):
        num_points = data.pos.shape[0]
        new_data = Data()
        for key in data.keys:
            if key == KDTREE_KEY:
                continue
            item = data[key]
            if torch.is_tensor(item) and num_points == item.shape[0]:
                item = item[self._indices].clone()
            elif torch.is_tensor(item):
                item = item.clone()
            setattr(new_data, key, item)
        return new_data


class CylinderNormalizeScale(object):
    """ Normalize points within a cylinder

    """

    def __init__(self, normalize_z=True):
        self._normalize_z = normalize_z

    def _process(self, data):
        data.pos -= data.pos.mean(dim=0, keepdim=True)
        scale = (1 / data.pos[:, :-1].abs().max()) * 0.999999
        data.pos[:, :-1] = data.pos[:, :-1] * scale
        if self._normalize_z:
            scale = (1 / data.pos[:, -1].abs().max()) * 0.999999
            data.pos[:, -1] = data.pos[:, -1] * scale
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(normalize_z={})".format(self.__class__.__name__, self._normalize_z)


class RandomSymmetry(object):
    """ Apply a random symmetry transformation on the data

    Parameters
    ----------
    axis: Tuple[bool,bool,bool], optional
        axis along which the symmetry is applied
    """

    def __init__(self, axis=[False, False, False]):
        self.axis = axis

    def __call__(self, data):

        for i, ax in enumerate(self.axis):
            if ax:
                if torch.rand(1) < 0.5:
                    c_max = torch.max(data.pos[:, i])
                    data.pos[:, i] = c_max - data.pos[:, i]
        return data

    def __repr__(self):
        return "Random symmetry of axes: x={}, y={}, z={}".format(*self.axis)


class RandomNoise(object):
    """ Simple isotropic additive gaussian noise (Jitter)

    Parameters
    ----------
    sigma:
        Variance of the noise
    clip:
        Maximum amplitude of the noise
    """

    def __init__(self, sigma=0.01, clip=0.05, p: float = None):
        self.sigma = sigma
        self.clip = clip
        self.p = 1 if p is None else p

    def __call__(self, data):
        if random.random() < self.p:
            noise = self.sigma * torch.randn(data.pos.shape)
            noise = noise.clamp(-self.clip, self.clip)
            data.pos = data.pos + noise
        return data

    def __repr__(self):
        return "{}(sigma={}, clip={})".format(self.__class__.__name__, self.sigma, self.clip)


class StatZOutlierRemoval:
    def __init__(self, threshold: float = 4, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.threshold = threshold  # std deviation

    def __call__(self, data):
        z = data.pos[:, 2]
        m = z.mean()
        s = z.std()
        out = abs((z - m) / s)
        mask = out < self.threshold
        data = apply_mask(data, mask, self.skip_list)
        return data

    def __repr__(self):
        return "{}(threshold={})".format(self.__class__.__name__, self.p)


class DBSCANZOutlierRemoval:
    def __init__(self, eps: float = 1, min_samples: int = 10, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = DBSCAN1D(eps=eps, min_samples=min_samples)

    def __call__(self, data):
        z = data.pos[:, 2]
        label = torch.tensor(self.dbscan.fit_predict(z[:, None]))
        mask = label != -1
        mask = (z <= z[mask].max()) & (z >= z[mask].min())
        data = apply_mask(data, mask, self.skip_list)
        return data

    def __repr__(self):
        return "{}(eps={},min_samples={})".format(self.__class__.__name__, self.eps, self.min_samples)


class OPTICSZOutlierRemoval:
    def __init__(self, eps: float = 1, min_samples: int = 10, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan = OPTICS(eps=eps, min_samples=min_samples, cluster_method="dbscan")

    def __call__(self, data):
        z = data.pos[:, 2]
        label = torch.tensor(self.dbscan.fit_predict(z[:, None]))
        mask = label != -1
        mask = (z <= z[mask].max()) & (z >= z[mask].min())
        # if ~(mask).any():
        #     from openpoints.dataset import vis_points
        #     vis_points(data['pos'], mask)
        data = apply_mask(data, mask, self.skip_list)
        return data

    def __repr__(self):
        return "{}(eps={},min_samples={})".format(self.__class__.__name__, self.eps, self.min_samples)


class KernelDensityZOutlierRemoval:
    def __init__(self, bandwidth: float = 1, p: float = 0.05, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.bandwidth = bandwidth
        self.p = p
        self.kd = KernelDensity(kernel="gaussian", bandwidth=bandwidth)

    def __call__(self, data):
        z = data.pos[:, 2]
        label = torch.tensor(self.kd.fit(z[:, None]).score_samples(z[:, None]))
        mask = label > np.log(self.p)
        mask = (z <= z[mask].max()) & (z >= z[mask].min())
        # if ~(mask).any():
        #     from openpoints.dataset import vis_points
        #     vis_points(data['pos'], mask)
        data = apply_mask(data, mask, self.skip_list)
        return data

    def __repr__(self):
        return "{}(bandwidth={},p={})".format(self.__class__.__name__, self.bandwidth, self.p)


class ScalePos:
    def __init__(self, scale_x=1., scale_y=1., scale_z=1., op="mul"):
        self.scale = torch.tensor([scale_x, scale_y, scale_z]).unsqueeze(0)
        self.op_str = op
        self.op = torch.mul if op == "mul" else torch.div

    def __call__(self, data):
        data.pos = self.op(data.pos, self.scale)
        return data

    def __repr__(self):
        return "{}(scale={},op={})".format(self.__class__.__name__, self.scale, self.op_str)


def maxmin_center(data):
    return (data.pos.amax(dim=0, keepdim=True) + data.pos.amin(dim=0, keepdim=True)) / 2.


def quantile_center(data):
    return (torch.quantile(data.pos, 0.99, dim=0, keepdim=True) +
            torch.quantile(data.pos, 0.01, dim=0, keepdim=True)) / 2.


def mean_center(data):
    return data.pos.mean(axis=0, keepdims=True)


class CenterPosPerSample:
    r"""Centers point positions by a defined 'center' function.
        Parameters
    -----------
    center_x: bool
        centering the x-axis.
    center_y: bool
        centering the y-axis.
    center_z: bool
        centering the z-axis.
    center: str
        which center function is used (choose from: 'mean', 'quantile', 'maxmin').
    """

    def __init__(self, center_x: bool = True, center_y: bool = True, center_z: bool = False, center: str = "mean"):
        self.center_ = torch.FloatTensor([[center_x, center_y, center_z]])
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.center_any = center_x or center_z or center_y
        self.center = center
        if center == "mean":
            self.agg = mean_center
        elif center == "quantile":
            self.agg = quantile_center
        elif center == "maxmin":
            self.agg = maxmin_center
        else:
            raise Exception(f"Unknown center function: {center} (should be 'mean', 'quantile', or 'maxmin')")

    def __call__(self, data):
        if self.center_any:
            center = self.agg(data) * self.center_
            data.pos -= center
        return data

    def __repr__(self):
        return "{}(center_x={},center_y={},center_z={},center={})".format(
            self.__class__.__name__, self.center_x, self.center_y, self.center_z, self.center
        )


class CenterXYbyZ:
    r"""Centers xy point positions by z selected points.
        Parameters
    -----------
    center_x: float
        centering the x-axis.
    center_y: float
        centering the y-axis.
    z_thresh_min: float
        min threshold for selecting z.
    z_thresh_max: float
        max threshold for selecting z.
    """

    def __init__(self, center_x: float = 0., center_y: float = 0., z_thresh_min: float = 0., z_thresh_max: float = 1.):
        self.z_thresh_min = z_thresh_min
        self.z_thresh_max = z_thresh_max
        self.center_ = torch.FloatTensor([[center_x, center_y]])

    def __call__(self, data):
        z_points = (self.z_thresh_min < data.pos[:, 2]) & (data.pos[:, 2] < self.z_thresh_max)
        pos = data.pos[:, :2]
        amax = pos[z_points].amax(0, keepdim=True)
        amin = pos[z_points].amin(0, keepdim=True)
        pos -= (amax + amin) / 2.
        pos += self.center_
        data.pos[:, :2] = pos
        data["pos_deviation"] = amax - amin
        data["pos_center_points"] = z_points.sum()
        return data

    def __repr__(self):
        return "{}(center_x={},center_y={},z_thresh_min={},z_thresh_max={})".format(
            self.__class__.__name__, self.center_[0, 0], self.center_[0, 1], self.z_thresh_min, self.z_thresh_max
        )


class FixedCenterPosPerSample:
    r"""Centers point positions by a defined 'center' function.
        Parameters
    -----------
    center_x: float
        centering the x-axis.
    center_y: float
        centering the y-axis.
    center_z: float
        centering the z-axis.
    """

    def __init__(self, center_x: float = 0.5, center_y: float = 0.5, center_z: float = 0.5):
        self.center_ = torch.FloatTensor([[center_x, center_y, center_z]])

    def __call__(self, data):
        data.pos -= (data.pos.amax(0, keepdim=True) + data.pos.amin(0, keepdim=True)) / 2.
        data.pos += self.center_
        return data

    def __repr__(self):
        return "{}(center_x={},center_y={},center_z={})".format(
            self.__class__.__name__, self.center_[0, 0], self.center_[0, 1], self.center_[0, 2],
        )


class MoveCenterPosPerSample:
    r"""Centers point positions by a defined 'center' function.
        Parameters
    -----------
    center_x: float
        centering the x-axis.
    center_y: float
        centering the y-axis.
    center_z: float
        centering the z-axis.
    """

    def __init__(self, center_x: float = 0.5, center_y: float = 0.5, center_z: float = 0.5):
        self.center_ = torch.FloatTensor([[center_x, center_y, center_z]])

    def __call__(self, data):
        data.pos += self.center_
        return data

    def __repr__(self):
        return "{}(center_x={},center_y={},center_z={})".format(
            self.__class__.__name__, self.center_[0, 0], self.center_[0, 1], self.center_[0, 2],
        )


class RandomShiftPos:
    def __init__(self, max_x: float = 0.01, max_y: float = 0.01, max_z: float = 0.01, p: float = 0.5):
        self.max_ = torch.FloatTensor([[max_x, max_y, max_y]])
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            data.pos += (torch.rand(1, 3) * 2 * self.max_) - self.max_
        return data

    def __repr__(self):
        return "{}(max_x={},max_y={},max_z={},p={})".format(
            self.__class__.__name__, self.max_x, self.max_y, self.max_z, self.p
        )


class StartZFromZero:
    def __call__(self, data):
        data.pos[:, 2] -= data.pos[:, 2].min()
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__, )


class AddRandomPoints:
    r"""Add points randomly within existing cloud bounds. Only works without additional features.
    Intended for regression or classification (will not add point-wise labels).
    Parameters
    -----------
    n_max_points: int
        Maximal total number of points (will not add points if there are already many).
    add_ratio_min: float
        Minimal amount of points to add according to existing number of points.
    add_ratio_max: float
        Maximal amount of points to add according to existing number of points.

    """

    def __init__(self, n_max_points: int, add_ratio_min: float, add_ratio_max: float, p: float = 0.5):
        self.n_max_points = n_max_points
        self.add_ratio_min = add_ratio_min
        self.add_ratio_max = add_ratio_max
        self.p = p

    def __call__(self, data):
        n_ori_points = len(data.pos)
        if n_ori_points >= self.n_max_points:
            return data

        if self.p > random.random():
            ratio = random.random() * (self.add_ratio_max - self.add_ratio_min) + self.add_ratio_min
            n_points = int(ratio * n_ori_points)
            n_points += np.amin([0, self.n_max_points - (n_ori_points + n_points)])  # remove points if necessary

            min_ = data.pos.amin(0, keepdim=True)
            max_ = data.pos.amin(0, keepdim=True)
            random_points = (torch.rand(n_points, data.pos.shape[1]) * (max_ - min_) + min_)

            data.pos = torch.cat([data.pos, random_points], 0)
        return data

    def __repr__(self):
        return "{}(n_max_points={},add_ratio_min={},add_ratio_max={},p={})".format(
            self.__class__.__name__, self.n_max_points, self.add_ratio_min, self.add_ratio_max, self.p
        )


class CopyJitterRandomPoints:
    r"""Randomly copies and jitters points. Will also copy features and labels (if present) but not alter them.
    Parameters
    -----------
    n_max_points: int
        Maximal total number of points (will not add points if there are already many).
    add_ratio_min: float
        Minimal amount of points to add according to existing number of points.
    add_ratio_max: float
        Maximal amount of points to add according to existing number of points.
    sigma:
        Variance of the noise
    clip:
        Maximum amplitude of the noise


    """

    def __init__(self, n_max_points: int, add_ratio_min: float, add_ratio_max: float,
                 sigma: float, clip: float, p: float = 0.5):
        self.n_max_points = n_max_points
        self.add_ratio_min = add_ratio_min
        self.add_ratio_max = add_ratio_max
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, data):
        n_ori_points = len(data.pos)
        if n_ori_points >= self.n_max_points:
            return data

        if self.p > random.random():
            ratio = random.random() * (self.add_ratio_max - self.add_ratio_min) + self.add_ratio_min
            n_points = int(ratio * n_ori_points)
            n_points += np.amin([0, self.n_max_points - (n_ori_points + n_points)])  # remove points if necessary

            idx = np.random.choice(n_ori_points, size=n_points, replace=True)
            random_points = data.pos[idx].clone()
            noise = self.sigma * torch.randn(random_points.shape)
            noise = noise.clamp(-self.clip, self.clip)
            random_points += noise

            if data.x is not None:
                data.x = torch.cat([data.x, data.x[idx].clone()], 0)
            if data.y is not None and len(data.y) == len(data.pos):
                data.y = torch.cat([data.y, data.y[idx].clone()], 0)

            data.pos = torch.cat([data.pos, random_points], 0)
        return data

    def __repr__(self):
        return "{}(n_max_points={},add_ratio_min={},add_ratio_max={},sigma={},clip={},p={})".format(
            self.__class__.__name__, self.n_max_points, self.add_ratio_min, self.add_ratio_max,
            self.sigma, self.clip, self.p
        )


class RandomScaling:
    r""" Scales node positions by a randomly sampled factor ``s1, s2, s3`` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \left[
        \begin{array}{ccc}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{array}
        \right]


    for three-dimensional positions.

    Parameters
    -----------
    scales:
        scaling factor interval, e.g. ``(a, b)``, then scale \
        is randomly sampled from the range \
        ``a <=  b``. \
    """

    def __init__(self, scales=None):
        assert is_iterable(scales) and len(scales) == 2
        assert scales[0] <= scales[1]
        self.scales = scales

    def __call__(self, data):
        scale = self.scales[0] + torch.rand((3,)) * (self.scales[1] - self.scales[0])
        data.pos = data.pos * scale
        if getattr(data, "norm", None) is not None:
            data.norm = data.norm / scale
            data.norm = torch.nn.functional.normalize(data.norm, dim=1)
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.scales)


class MeshToNormal(object):
    """ Computes mesh normals (IN PROGRESS)
    """

    def __init__(self):
        pass

    def __call__(self, data):
        if hasattr(data, "face"):
            pos = data.pos
            face = data.face
            vertices = [pos[f] for f in face]
            normals = torch.cross(vertices[0] - vertices[1], vertices[0] - vertices[2], dim=1)
            normals = F.normalize(normals)
            data.normals = normals
        return data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class MultiScaleTransform(object):
    """ Pre-computes a sequence of downsampling / neighboorhood search on the CPU.
    This currently only works on PARTIAL_DENSE formats

    Parameters
    -----------
    strategies: Dict[str, object]
        Dictionary that contains the samplers and neighbour_finder
    """

    def __init__(self, strategies):
        self.strategies = strategies
        self.num_layers = len(self.strategies["sampler"])

    @staticmethod
    def __inc__wrapper(func, special_params):
        def new__inc__(key, num_nodes, special_params=None, func=None):
            if key in special_params:
                return special_params[key]
            else:
                return func(key, num_nodes)

        return partial(new__inc__, special_params=special_params, func=func)

    def __call__(self, data: Data) -> MultiScaleData:
        # Compute sequentially multi_scale indexes on cpu
        data.contiguous()
        ms_data = MultiScaleData.from_data(data)
        precomputed = [Data(pos=data.pos)]
        upsample = []
        upsample_index = 0
        for index in range(self.num_layers):
            sampler, neighbour_finder = self.strategies["sampler"][index], self.strategies["neighbour_finder"][index]
            support = precomputed[index]
            new_data = Data(pos=support.pos)
            if sampler:
                query = sampler(new_data.clone())
                query.contiguous()

                if len(self.strategies["upsample_op"]):
                    if upsample_index >= len(self.strategies["upsample_op"]):
                        raise ValueError("You are missing some upsample blocks in your network")

                    upsampler = self.strategies["upsample_op"][upsample_index]
                    upsample_index += 1
                    pre_up = upsampler.precompute(query, support)
                    upsample.append(pre_up)
                    special_params = {}
                    special_params["x_idx"] = query.num_nodes
                    special_params["y_idx"] = support.num_nodes
                    setattr(pre_up, "__inc__", self.__inc__wrapper(pre_up.__inc__, special_params))
            else:
                query = new_data

            s_pos, q_pos = support.pos, query.pos
            if hasattr(query, "batch"):
                s_batch, q_batch = support.batch, query.batch
            else:
                s_batch, q_batch = (
                    torch.zeros((s_pos.shape[0]), dtype=torch.long),
                    torch.zeros((q_pos.shape[0]), dtype=torch.long),
                )

            idx_neighboors = neighbour_finder(s_pos, q_pos, batch_x=s_batch, batch_y=q_batch)
            special_params = {}
            special_params["idx_neighboors"] = s_pos.shape[0]
            setattr(query, "idx_neighboors", idx_neighboors)
            setattr(query, "__inc__", self.__inc__wrapper(query.__inc__, special_params))
            precomputed.append(query)
        ms_data.multiscale = precomputed[1:]
        upsample.reverse()  # Switch to inner layer first
        ms_data.upsample = upsample
        return ms_data

    def __repr__(self):
        return "{}".format(self.__class__.__name__)


class ShuffleData(object):
    """ This transform allow to shuffle feature, pos and label tensors within data
    """

    def _process(self, data):
        return shuffle_data(data)

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in tq(data)]
            data = list(chain(*data))  # 2d list needs to be flatten
        else:
            data = self._process(data)
        return data


class ShiftVoxels:
    """ Trick to make Sparse conv invariant to even and odds coordinates
    https://github.com/chrischoy/SpatioTemporalSegmentation/blob/master/lib/train.py#L78

    Parameters
    -----------
    apply_shift: bool:
        Whether to apply the shift on indices
    """

    def __init__(self, apply_shift=True, p=0.5):
        self._apply_shift = apply_shift
        self.p = p

    def __call__(self, data):
        if self._apply_shift and random.random() < self.p:
            if not hasattr(data, "coords"):
                raise Exception("should quantize first using GridSampling3D")

            if not isinstance(data.coords, torch.IntTensor):
                raise Exception("The pos are expected to be coordinates, so torch.IntTensor")
            data.coords[:, :3] += (torch.rand(3) * 100).type_as(data.coords)
        return data

    def __repr__(self):
        return "{}(apply_shift={})".format(self.__class__.__name__, self._apply_shift)


class RandomDropout:
    """ Randomly drop points from the input data

    Parameters
    ----------
    dropout_ratio : float, optional
        Ratio that gets dropped
    dropout_application_ratio   : float, optional
        chances of the dropout to be applied
    """

    def __init__(self, dropout_ratio: float = 0.2, dropout_application_ratio: float = 0.5, min_points: int = 0,
                 skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio
        self.min_points = min_points

    def __call__(self, data):
        N = len(data.pos)
        if N > self.min_points and random.random() < self.dropout_application_ratio:
            data = FixedPointsOwn(int(N * (1 - self.dropout_ratio)), skip_list=self.skip_list)(data)
        return data

    def __repr__(self):
        return "{}(dropout_ratio={}, dropout_application_ratio={})".format(
            self.__class__.__name__, self.dropout_ratio, self.dropout_application_ratio
        )


def apply_mask(data, mask, skip_keys=[]):
    size_pos = len(data.pos)
    for k in data.keys:
        if torch.is_tensor(data[k]) and size_pos == len(data[k]) and k not in skip_keys:
            data[k] = data[k][mask]
    return data


@numba.jit(nopython=True, cache=True)
def rw_mask(pos, ind, dist, mask_vertices, random_ratio=0.04, num_iter=5000):
    rand_ind = np.random.randint(0, len(pos))
    for _ in range(num_iter):
        mask_vertices[rand_ind] = False
        if np.random.rand() < random_ratio:
            rand_ind = np.random.randint(0, len(pos))
        else:
            neighbors = ind[rand_ind][dist[rand_ind] > 0]
            if len(neighbors) == 0:
                rand_ind = np.random.randint(0, len(pos))
            else:
                n_i = np.random.randint(0, len(neighbors))
                rand_ind = neighbors[n_i]
    return mask_vertices


def topview_sample(data, num_samples: int):
    # simulates a little airborne lidar behavior (discarding of lower points more likely)
    num_nodes = data.num_nodes
    z = data.pos[:, 2].numpy()
    choice = random.choices(np.arange(num_nodes), weights=z, k=num_samples)

    for key, item in data:
        if key == 'num_nodes':
            data.num_nodes = choice.size(0)
        elif (torch.is_tensor(item) and item.size(0) == num_nodes
              and item.size(0) != 1):
            data[key] = item[choice]

    return data


class RandomGroundRemoval:
    def __init__(self, min_v: float, max_v: float, p: float = 0.5, min_points: int = 500, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)
        self.min_v = min_v
        self.max_v = max_v
        self.range = max_v - min_v
        self.p = p
        self.min_points = min_points

    def __call__(self, data):
        if random.random() < self.p:
            pos = data.pos
            remove_v = random.random() * self.range + self.min_v
            cond = pos[:, 2] > remove_v
            if cond.sum() < self.min_points:
                return data
            pos[:, 2] -= remove_v
            data = apply_mask(data, cond, self.skip_list)

        return data


class RadiusObjectAdder:
    def __init__(self, areas, root_folder: str, dataset_name: str, processed_folder: str,
                 min_radius: float, max_radius: float, n_max_objects,
                 rot_x: float, rot_y: float, rot_z: float, indicator_key: str = None,
                 adjust_point_density: bool = False, density_topview_sample: bool = False, density_index: int = 0,
                 density_adjustment: list = 1., split: str = "train", zero_center_z: bool = False,
                 only_doubled_batch: bool = False, in_memory: bool = False, p: float = 0.5):
        areas = OmegaConf.to_container(areas)
        self.areas = {area: areas[area] for area in areas if areas[area]["type"] == "object"}
        self.processed_dir = PPath(os.path.join(root_folder, dataset_name, processed_folder, split))
        self.object_files = list(chain(*[glob(str(self.processed_dir / f"{area}/*.pt")) for area in self.areas]))

        self.min_radius = min_radius
        self.max_radius = max_radius
        self.adjust_point_density = adjust_point_density
        if adjust_point_density:
            self.density_index = density_index
            self.density_topview_sample = density_topview_sample
            # adjust original data density given as range
            # (e.g., 0.5 will decrease original point density,
            # thus potentially removing more points from the added object)
            self.density_adjustment = (density_adjustment[0], density_adjustment[1])
        self.memory = {}
        self.in_memory = in_memory
        self.random_rotation = Random3AxisRotation(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
        if isinstance(n_max_objects, int):
            n_max_objects = {
                "object": n_max_objects,
                "scene": n_max_objects,
            }
        self.n_max_objects: dict = n_max_objects
        self.p: float = p
        self.zero_center_z = zero_center_z
        self.indicator_key = indicator_key
        self.only_doubled_batch = only_doubled_batch

    def __call__(self, data):
        if len(self.object_files) == 0:
            self.object_files = list(chain(*[glob(str(self.processed_dir / f"{area}/*.pt")) for area in self.areas]))
            assert len(self.object_files) > 0, "no objects given for RadiusObjectAdder"
        ori_n = None
        if random.random() < self.p and (
                not self.only_doubled_batch or (self.only_doubled_batch and data.get("is_double", False))):
            sample_type = "object" if data.area_name in self.areas else "scene"
            n_objects = random.randint(1, self.n_max_objects[sample_type])
            files = np.random.choice(self.object_files, n_objects, replace=True)
            pos_ = []
            feat_ = []
            i = 0
            while i < len(files):
                file = files[i]
                i += 1
                if self.in_memory:
                    new_object = self.memory.get(file, None)
                    if new_object is None:
                        new_object = torch.load(file)
                        self.memory[file] = new_object.clone()
                    else:
                        new_object = new_object.clone()
                else:
                    new_object = torch.load(file)
                if self.zero_center_z:
                    new_object.pos[:, 2] -= new_object.pos[:, 2].min()
                new_object = self.random_rotation(new_object)

                if self.adjust_point_density:
                    # only removes points if too dense, will not add points
                    sample_density = data["local_stats"][self.density_index]
                    obj_density = new_object["local_stats"][self.density_index]
                    density_adjustment_factor = random.random()
                    density_adjustment_factor *= self.density_adjustment[1] - self.density_adjustment[0]
                    density_adjustment_factor += self.density_adjustment[0]
                    drop_ratio = (sample_density * density_adjustment_factor) / obj_density
                    if drop_ratio < 1:
                        if self.density_topview_sample:
                            new_object = topview_sample(new_object, int(drop_ratio * len(new_object.pos)))
                        else:
                            new_object = FP(int(drop_ratio * len(new_object.pos)), replace=False)(new_object)

                # random point in outer circle
                angle = random.uniform(0, 2 * math.pi)

                min_radius = self.min_radius
                max_radius = self.max_radius
                # add safety margin if we are given center deviation
                if "pos_deviation" in new_object:
                    min_radius += (new_object["pos_deviation"] ** 2).sum() ** .5 / 2  # pythagoras
                    if min_radius > max_radius:  # add another object to list and skip this one
                        files = np.concatenate([files, np.random.choice(self.object_files, 1)], axis=0)
                        continue
                radius = random.uniform(min_radius, max_radius)
                shift = torch.tensor(([[math.cos(angle), math.sin(angle), 0]])) * radius  # no shift in z

                pos_.append(new_object.pos + shift)
                feat_.append(new_object.x)

            ori_n = len(data.pos)
            data.pos = torch.cat([data.pos, *pos_], 0)
            if data.x is not None:
                if len(feat_) > 0 and feat_[0] is not None:
                    data.x = torch.cat([data.x, *feat_], 0)
                else:
                    data.x = torch.cat([data.x, torch.zeros(len(data.pos) - ori_n, data.x.shape[1])], 0)

        if self.indicator_key is not None:
            if ori_n is not None:
                indicator = torch.zeros(len(data.pos))
                indicator[ori_n:] = True
            else:
                indicator = torch.zeros(len(data.pos))

            data[self.indicator_key] = indicator
        return data


class CubeCrop(object):
    """
    Crop cubically the point cloud. This function take a cube of size c
    centered on a random point, then points outside the cube are rejected.

    Parameters
    ----------
    c: float, optional
        half size of the cube
    rot_x: float_otional
        rotation of the cube around x axis
    rot_y: float_otional
        rotation of the cube around x axis
    rot_z: float_otional
        rotation of the cube around x axis
    """

    def __init__(
            self, c: float = 1, rot_x: float = 180, rot_y: float = 180, rot_z: float = 180,
            grid_size_center: float = 0.01
    ):
        self.c = c
        self.random_rotation = Random3AxisRotation(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)
        self.grid_sampling = GridSampling3D(grid_size_center, mode="last")

    def __call__(self, data):
        data_c = self.grid_sampling(data.clone())
        data_temp = data.clone()
        i = torch.randint(0, len(data_c.pos), (1,))
        center = data_c.pos[i]
        min_square = center - self.c
        max_square = center + self.c
        data_temp.pos = data_temp.pos - center
        data_temp = self.random_rotation(data_temp)
        data_temp.pos = data_temp.pos + center
        mask = torch.prod((data_temp.pos - min_square) > 0, dim=1) * torch.prod((max_square - data_temp.pos) > 0, dim=1)
        mask = mask.to(torch.bool)
        data = apply_mask(data, mask)
        return data

    def __repr__(self):
        return "{}(c={}, rotation={})".format(self.__class__.__name__, self.c, self.random_rotation)


class FixedPointsOwn(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud (functional name: :obj:`fixed_points`).

    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples points
            without replacement. (default: :obj:`True`)
        allow_duplicates (bool, optional): In case :obj:`replace` is
            :obj`False` and :obj:`num` is greater than the number of points,
            this option determines whether to add duplicated nodes to the
            output points or not.
            In case :obj:`allow_duplicates` is :obj:`False`, the number of
            output points might be smaller than :obj:`num`.
            In case :obj:`allow_duplicates` is :obj:`True`, the number of
            duplicated points are kept to a minimum. (default: :obj:`False`)
    """

    def __init__(self, num, replace=False, allow_duplicates=True, skip_list: list = None):
        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list) if isinstance(skip_list,
                                                                                                   OmegaConf) else skip_list
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data):
        num_nodes = data.num_nodes

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = choice.size(0)
            elif bool(re.search('edge', key)):
                continue
            elif (torch.is_tensor(item) and item.size(0) == num_nodes and key not in self.skip_list
                  and (item.size(0) != 1) or key == "pos"):
                data[key] = item[choice]
        assert data.pos.shape[
                   0] == self.num, f"pos: {data.pos.shape}, y: {data.y_mol.shape}, {data.y_mol_mask.shape}, choice: {len(choice)} {self.num}"
        return data


class CylinderExtend(object):
    """
    Restrict extend the point cloud to a cylinder. This function take a radius
    centered at the origin, then points outside are rejected.
    Parameters
    ----------
    radius: float
        half size of the x axis of the rectangle
    skip_list: list
        list of keys not to mask away
    """

    def __init__(self, radius: float, skip_list: list = None):
        self.radius = radius
        self.skip_list = skip_list

    def __call__(self, data):
        pos = data.pos
        if not hasattr(data, KDTREE_KEY):
            tree = KDTree(np.asarray(pos[:, :-1]), leaf_size=50)
            setattr(data, KDTREE_KEY, tree)
        else:
            tree = getattr(data, KDTREE_KEY)
        idx = tree.query_radius([[0., 0.]], self.radius)[0]
        mask = torch.zeros(len(pos)).bool()
        mask[idx] = True

        data = apply_mask(data, mask, self.skip_list)
        return data

    def __repr__(self):
        return "{}(radius={}, skip_list={})".format(self.__class__.__name__, self.radius, self.skip_list)


class RectangleExtend(object):
    """
    Restrict extend the point cloud to a rectangle. This function take a rectangle of size (e_x, e_y, e_z)
    centered at the origin, then points outside are rejected.
    Parameters
    ----------
    e_x: float, optional
        half size of the x axis of the rectangle
    e_y: float, optional
        half size of the y axis of the rectangle
    e_z: float, optional
        half size of the z axis of the rectangle
    """

    def __init__(self, e_x: float = 1, e_y: float = 1, e_z: float = 1, ):
        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z

    def __call__(self, data):
        pos = data.pos
        posx = pos[:, 0]
        posy = pos[:, 1]
        posz = pos[:, 2]
        mask = (posx < self.e_x) & (posx > -self.e_x) & \
               (posy < self.e_y) & (posx > -self.e_y) & \
               (posz < self.e_z) & (posz > -self.e_z)
        data = apply_mask(data, mask)
        return data

    def __repr__(self):
        return "{}(e_x={}, e_y={}, e_z={})".format(self.__class__.__name__, self.e_x, self.e_y, self.e_z)


def append_skeleton(self, data, skeleton):
    if self.cage_skeleton:
        min_z = data.pos[:, -1].min()
        max_z = data.pos[:, -1].max()
        heights = torch.arange(min_z, max_z + self.height_skeleton_pts, self.height_skeleton_pts).float()
        n_heights = len(heights)
        n_pts = len(skeleton)
        skeleton = skeleton.repeat_interleave(n_heights, 0)
        skeleton[:, 2] *= heights.reshape(-1).repeat(n_pts)
    else:

        skeleton *= self.height_skeleton_pts
    num_skeleton_pts = len(skeleton)
    indicator = torch.zeros(len(data.pos) + num_skeleton_pts)
    indicator[-num_skeleton_pts:] = 1.0
    # add empty features for skeleton
    size_pos = len(data.pos)
    for k in data.keys:
        if torch.is_tensor(data[k]) and size_pos == len(data[k]) and k not in ["pos"] + self.skip_list:
            dtype = data[k].dtype
            if len(data[k].shape) > 1:
                n_feat = data[k].shape[1]
                data[k] = torch.cat([data[k], torch.ones(num_skeleton_pts, n_feat, dtype=dtype)], 0)
            else:
                data[k] = torch.cat([data[k], torch.ones(num_skeleton_pts, dtype=dtype)], 0)
    data["skeleton"] = indicator
    data["pos"] = torch.cat([data["pos"], skeleton], 0)


class Polygon2dExtend(object):
    """
    Restrict extend the point cloud to a given polygon. This function takes point tuples of size
    (e.g., [[0, 1], [1, 0], [1, 1]]).
    centered at the origin, then points outside are rejected.

    Parameters
    ----------
    polygon: list
        List of tuples containing the border points of the polygon
    """

    def __init__(self, polygon, skip_list: list = None, add_skeleton_pts: bool = False,
                 num_skeleton_pts: int = 100, height_skeleton_pts: float = 1.0,
                 cage_skeleton: bool = False):
        self.polygon = Path(polygon)

        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)

        self.add_skeleton_pts = add_skeleton_pts
        self.num_skeleton_pts = num_skeleton_pts
        self.height_skeleton_pts = height_skeleton_pts
        self.cage_skeleton = cage_skeleton

        if add_skeleton_pts:
            skeleton = torch.tensor(self.polygon.interpolated(self.num_skeleton_pts).vertices).float()
            self.skeleton = torch.cat([skeleton, torch.ones(len(skeleton), 1)], 1)

    def __call__(self, data):
        pos = data.pos[:, [0, 1]]
        mask = self.polygon.contains_points(pos)
        data = apply_mask(data, mask, self.skip_list)
        if self.add_skeleton_pts:
            append_skeleton(self, data, self.skeleton)

        return data

    def __repr__(self):
        return "{}(polygon={})".format(self.__class__.__name__, self.polygon.to_polygons())


class RandomPolygon2dExtend(object):
    """
    Restrict extend the point cloud to a given polygon. This function takes point tuples of size
    (e.g., [[0, 1], [1, 0], [1, 1]]).
    centered at the origin, then points outside are rejected.

    Parameters
    ----------
    polygons: list
        List of polygons, each defined by tuples containing the border points
    """

    def __init__(self, polygons: list, skip_list: list = None, size_min: float = 1, size_max: float = 1,
                 rotate: float = 180,
                 add_skeleton_pts: bool = False, num_skeleton_pts: int = 100, height_skeleton_pts: float = 1.0,
                 cage_skeleton: bool = False):
        self.polygons = [polygon for polygon in polygons]
        self.n_p = len(self.polygons)
        self.size_min = size_min
        self.size_max = size_max
        self.rotate = rotate

        self.skip_list = [] if skip_list is None else OmegaConf.to_object(skip_list)

        self.add_skeleton_pts = add_skeleton_pts
        self.num_skeleton_pts = num_skeleton_pts
        self.height_skeleton_pts = height_skeleton_pts
        self.cage_skeleton = cage_skeleton

    def __call__(self, data):
        pos = data.pos[:, [0, 1]]
        polygon = self.polygons[np.random.choice(self.n_p)]
        if polygon != "None":
            rand_scale = np.random.rand() * (self.size_max - self.size_min) + self.size_min
            trans = (1 - rand_scale) / 2
            rand_rotate = np.random.rand() * self.rotate * np.sign(np.random.rand() - .5)
            A = Affine2D().scale(rand_scale).translate(trans, trans).rotate_deg_around(0.5, 0.5, rand_rotate)
            polygon = Path(polygon).transformed(A)
            mask = polygon.contains_points(pos)
            if mask.sum() > 0:  # apply masking if any points remain
                data = apply_mask(data, mask, self.skip_list)
            if self.add_skeleton_pts:
                skeleton = torch.tensor(polygon.interpolated(self.num_skeleton_pts).vertices).to(pos.dtype)
                skeleton = torch.cat([skeleton, torch.ones(len(skeleton), 1)], 1)

                append_skeleton(self, data, skeleton)
        elif self.add_skeleton_pts:
            data["skeleton"] = torch.zeros(len(data.pos), 1)
        return data

    def __repr__(self):
        return "{}(polygons={}, size_min={}, size_max={}, rotate={})".format(
            self.__class__.__name__, str(self.polygons), self.size_min, self.size_max, self.rotate
        )


class EllipsoidCrop(object):
    """

    """

    def __init__(
            self, a: float = 1, b: float = 1, c: float = 1, rot_x: float = 180, rot_y: float = 180, rot_z: float = 180
    ):
        """
        Crop with respect to an ellipsoid.
        the function of an ellipse is defined as:

        Parameters
        ----------
        a: float, optional
          half size of the cube
        b: float_otional
          rotation of the cube around x axis
        c: float_otional
          rotation of the cube around x axis


        """
        self._a2 = a ** 2
        self._b2 = b ** 2
        self._c2 = c ** 2
        self.random_rotation = Random3AxisRotation(rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)

    def _compute_mask(self, pos: torch.Tensor):
        mask = (pos[:, 0] ** 2 / self._a2 + pos[:, 1] ** 2 / self._b2 + pos[:, 2] ** 2 / self._c2) < 1
        return mask

    def __call__(self, data):
        data_temp = data.clone()
        i = torch.randint(0, len(data.pos), (1,))
        data_temp = self.random_rotation(data_temp)
        center = data_temp.pos[i]
        data_temp.pos = data_temp.pos - center
        mask = self._compute_mask(data_temp.pos)
        data = apply_mask(data, mask)
        return data

    def __repr__(self):
        return "{}(a={}, b={}, c={}, rotation={})".format(
            self.__class__.__name__, np.sqrt(self._a2), np.sqrt(self._b2), np.sqrt(self._c2), self.random_rotation
        )


class ZFilter(object):
    """
    Remove points lower or higher than certain values
    """

    def __init__(self, z_min, z_max, skip_keys: List = []):
        self.z_min = z_min
        self.z_max = z_max
        self.skip_keys = skip_keys

    def __call__(self, data):
        z = data.pos[:, 2]
        mask = (z > self.z_min) & (z < self.z_max)

        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(z_min={}, z_max={}, skip_keys={})".format(
            self.__class__.__name__, self.z_min, self.z_max, self.skip_keys
        )


class DensityFilter(object):
    """
    Remove points with a low density(compute the density with a radius search and remove points with)
    a low number of neighbors

    Parameters
    ----------
    radius_nn: float, optional
        radius for the neighbors search
    min_num: int, optional
        minimum number of neighbors to be dense
    skip_keys: int, optional
        list of attributes of data to skip when we apply the mask
    """

    def __init__(self, radius_nn: float = 0.04, min_num: int = 6, skip_keys: List = []):
        self.radius_nn = radius_nn
        self.min_num = min_num
        self.skip_keys = skip_keys

    def __call__(self, data):
        ind, dist = ball_query(data.pos, data.pos, radius=self.radius_nn, max_num=-1, mode=0)

        mask = (dist > 0).sum(1) > self.min_num
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(radius_nn={}, min_num={}, skip_keys={})".format(
            self.__class__.__name__, self.radius_nn, self.min_num, self.skip_keys
        )


class IrregularSampling(object):
    """
    a sort of soft crop. the more we are far from the center, the more it is unlikely to choose the point
    """

    def __init__(self, d_half=2.5, p=2, grid_size_center=0.1, skip_keys=[]):
        self.d_half = d_half
        self.p = p
        self.skip_keys = skip_keys
        self.grid_sampling = GridSampling3D(grid_size_center, mode="last")

    def __call__(self, data):
        data_temp = self.grid_sampling(data.clone())
        i = torch.randint(0, len(data_temp.pos), (1,))
        center = data_temp.pos[i]

        d_p = (torch.abs(data.pos - center) ** self.p).sum(1)

        sigma_2 = (self.d_half ** self.p) / (2 * np.log(2))
        thresh = torch.exp(-d_p / (2 * sigma_2))

        mask = torch.rand(len(data.pos)) < thresh
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(d_half={}, p={}, skip_keys={})".format(self.__class__.__name__, self.d_half, self.p, self.skip_keys)


class PeriodicSampling(object):
    """
    sample point at a periodic distance
    """

    def __init__(self, period=0.1, prop=0.1, box_multiplier=1, skip_keys=[]):
        self.pulse = 2 * np.pi / period
        self.thresh = np.cos(self.pulse * prop * period * 0.5)
        self.box_multiplier = box_multiplier
        self.skip_keys = skip_keys

    def __call__(self, data):
        data_temp = data.clone()
        max_p = data_temp.pos.max(0)[0]
        min_p = data_temp.pos.min(0)[0]

        center = self.box_multiplier * torch.rand(3) * (max_p - min_p) + min_p
        d_p = torch.norm(data.pos - center, dim=1)
        mask = torch.cos(self.pulse * d_p) > self.thresh
        data = apply_mask(data, mask, self.skip_keys)
        return data

    def __repr__(self):
        return "{}(pulse={}, thresh={}, box_mullti={}, skip_keys={})".format(
            self.__class__.__name__, self.pulse, self.thresh, self.box_multiplier, self.skip_keys
        )


class AddGround:
    '''simple class to add "n_points" ground points if less than "max_points" are present in a unit radius'''

    def __init__(self, max_points: int, n_points: int, xy_min: float = 0, xy_max: float = 1):
        self.max_points = max_points
        self.n_points = n_points
        self.xy_range = (xy_max - xy_min) / 2.
        self.xy_min = xy_min

    def __call__(self, data):
        nodes = data.num_nodes
        if nodes < self.max_points:
            data.pos = torch.rand(self.n_points, 3) * self.xy_range + self.xy_min
            data.pos[:, 2] = 0.0

        return data

    def __repr__(self):
        return "{}(max_points={}, n_points={})".format(
            self.__class__.__name__, self.max_points, self.n_points
        )


class MinPoints(FixedPointsOwn):
    r"""Samples a minimal number of :obj:`num` points and features from a point
    cloud.

    Args:
        num (int): The number of minimal points in point_idxs, resamples with replacement if less are present.
    """

    def __init__(self, num, skip_list: list = None):
        super().__init__(num, False, True, skip_list)

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes < self.num:
            # TODO verify state is persistent
            state = np.random.get_state()
            np.random.set_state(np.random.RandomState(42).get_state())
            data = super().__call__(data)
            np.random.set_state(state)
            return data

        return data

    def __repr__(self):
        return "{}(num={}, skip_list={})".format(
            self.__class__.__name__, self.num, self.skip_list
        )


class MaxPoints(FixedPointsOwn):
    r"""Samples a maximal number of :obj:`num` points and features from a point
    cloud.

    Args:
        num (int): The number to maximal number of points in point_idxs, resamples without replacement.
    """

    def __init__(self, num, skip_list: list = None):
        super().__init__(num, False, False, skip_list)

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes > self.num:
            # TODO verify state is persistent
            data = super().__call__(data)
            return data

        return data

    def __repr__(self):
        return "{}(num={}, skip_list={})".format(
            self.__class__.__name__, self.num, self.skip_list
        )
