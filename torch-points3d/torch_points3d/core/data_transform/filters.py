import random

import numpy as np
import torch

from torch_points3d.core.data_transform.features import PCACompute, compute_planarity


class FCompose(object):
    """
    allow to compose different filters using the boolean operation

    Parameters
    ----------
    list_filter: list
        list of different filter functions we want to apply
    boolean_operation: function, optional
        boolean function to compose the filter (take a pair and return a boolean)
    """

    def __init__(self, list_filter, boolean_operation=np.logical_and):
        self.list_filter = list_filter
        self.boolean_operation = boolean_operation

    def __call__(self, data):
        assert len(self.list_filter) > 0
        res = self.list_filter[0](data)
        for filter_fn in self.list_filter:
            res = self.boolean_operation(res, filter_fn(data))
        return res

    def __repr__(self):
        rep = "{}([".format(self.__class__.__name__)
        for filt in self.list_filter:
            rep = rep + filt.__repr__() + ", "
        rep = rep + "])"
        return rep


class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity of a pointcloud is above or below a threshold

    Parameters
    ----------
    thresh: float, optional
        threshold to filter low planar pointcloud
    is_leq: bool, optional
        choose whether planarity should be lesser or equal than the threshold or greater than the threshold.
    """

    def __init__(self, thresh=0.3, is_leq=True):
        self.thresh = thresh
        self.is_leq = is_leq

    def __call__(self, data):
        if getattr(data, "eigenvalues", None) is None:
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        if self.is_leq:
            return planarity <= self.thresh
        else:
            return planarity > self.thresh

    def __repr__(self):
        return "{}(thresh={}, is_leq={})".format(self.__class__.__name__, self.thresh, self.is_leq)


class RandomFilter(object):
    """
    Randomly select an elem of the dataset (to have smaller dataset) with a bernouilli distribution of parameter thresh.

    Parameters
    ----------
    thresh: float, optional
        the parameter of the bernouilli function
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        return random.random() < self.thresh

    def __repr__(self):
        return "{}(thresh={})".format(self.__class__.__name__, self.thresh)


class ClassificationFilter(object):
    """
    Select specific classes from "classification" feature to remove or keep.
    Keep is prioritized.

    Parameters
    ----------
    feature_index: int
        which index the classification is expected in
    class_indices:
        which class indices to select for keeping or removing
    keep: bool, optional
        keep the given class indices if true, else remove them (default: True)
    remove_feat: bool, optional
        if the feature should be removed after filtering (default: True)

    """

    def __init__(self, feature_index: int, class_indices: list, keep: bool = True, remove_feat: bool = True):
        self.class_indices = class_indices
        self.keep = keep
        self.feature_index = feature_index
        self.remove_feat = remove_feat

    def __call__(self, data):
        cls = data.x[:, self.feature_index]
        mask = torch.stack([cls == i for i in self.class_indices]).any(0)
        if not self.keep:
            mask = ~mask

        num_nodes = data.num_nodes
        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = mask.size(0)
            elif (torch.is_tensor(item) and item.size(0) == num_nodes
                  and item.size(0) != 1):
                data[key] = item[mask]

        if self.remove_feat:
            if data.x.shape[1] == 1:
                data.x = None
            else:
                data.x = torch.cat([data.x[:, :self.feature_index], data.x[:, self.feature_index + 1:]], 1)

        return data

    def __repr__(self):
        return "{}(feature_index={},class_indices={},keep={},remove_feat={})".format(
            self.__class__.__name__, self.feature_index, self.class_indices, self.keep, self.remove_feat
        )
