import logging
import os
from collections import OrderedDict
from functools import partial
from glob import glob
from itertools import chain, product
from pathlib import Path
from typing import Sized, Iterator

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj
import scipy.stats as scstats
import torch
from omegaconf import OmegaConf
from plyfile import PlyData
from shapely.geometry import Point
from sklearn.neighbors import KDTree
from torch.utils.data import Sampler
from torch_geometric.data import Dataset, Data
from tqdm.auto import tqdm

from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.metrics.instance_tracker import InstanceTracker
from torch_points3d.models import model_interface

log = logging.getLogger(__name__)


def read_pt(pt_file, feature_cols, delimiter: str):
    crs = None
    has_features = len(feature_cols) > 0
    if Path(pt_file).suffix in [".las", ".laz"]:
        backend = laspy.compression.LazBackend(0)
        if not backend.is_available():
            backend = laspy.compression.LazBackend(1)
            if not backend.is_available():
                backend = laspy.compression.LazBackend(2)
        loaded_file = laspy.read(pt_file, laz_backend=backend)
        pos = np.stack([loaded_file.x, loaded_file.y, loaded_file.z], 1)
        if has_features:
            features = np.stack([getattr(loaded_file, feature) for feature in feature_cols], 1)
        else:
            features = None

        # get crs
        for vlr in loaded_file.header.vlrs:
            if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
                # read general CRS (ignores specific parameters)
                crs = pyproj.CRS(vlr.string)
    elif Path(pt_file).suffix in [".ply"]:
        loaded_file = PlyData.read(pt_file)
        pos = np.stack([loaded_file.elements[0]["x"], loaded_file.elements[0]["y"], loaded_file.elements[0]["z"]], 1)
        if has_features:
            features = np.stack([loaded_file.elements[0][feat] for feat in feature_cols], 1)
        else:
            features = None
    else:
        # try to read as csv
        loaded_file = pd.read_csv(
            pt_file, header=None, engine="pyarrow", delimiter=delimiter, dtype=np.float32, skip_blank_lines=True
        )
        pos = loaded_file.values[:, :3]  # assumes first 3 values are positions
        if has_features:
            features = loaded_file[feature_cols]
        else:
            features = None

    return pos, features, crs


class Las(Dataset):
    """loads all las files into memory and creates samples based on a label_df"""

    def __init__(
            self, root, areas: dict, split: str, stats=None,
            xy_radius=15.,
            transform=None, targets=None, feature_cols=None, feature_scaling_dict: dict = None,
            pre_transform=None, pre_filter=None, save_local_stats: bool = True,
            min_pts_outer: int = 500, min_pts_inner: int = 250,
            save_processed: bool = True, processed_folder="processed", in_memory: bool = False,
            pos_dict: dict = None, features_dict: dict = None, pos_tree_dict: dict = None, crs_dict: dict = None
    ):
        self.root = root
        self.split = split

        self.min_pts_outer = min_pts_outer
        self.min_pts_inner = min_pts_inner

        # useful for double batch detection
        self.prev_idx = None

        assert save_processed or in_memory, "Samples are neither saved to processed folder or kept in memory! " \
                                            "(set either save_processed or in_memory to True)"
        self.in_memory = in_memory
        if in_memory:
            self.memory = {}

        if not save_processed and in_memory:
            log.info("Not saving any samples, storing areas in memory if not present on disk")

        self.save_processed = save_processed
        self.processed_folder = processed_folder

        self.save_local_stats = save_local_stats

        if pos_dict is not None or pos_tree_dict is not None:
            assert pos_dict is not None and pos_tree_dict is not None, \
                "if any of pos or pos_tree are given, both need to be there"

            assert (len(feature_cols) > 0 and (features_dict is not None)) or len(feature_cols) == 0, \
                "need to give features, if pos is given and there are features"
        self.pos_dict = {} if pos_dict is None else pos_dict
        self.features_dict = {} if features_dict is None else features_dict
        self.pos_tree_dict = {} if pos_tree_dict is None else pos_tree_dict
        self.crs_dict = {} if crs_dict is None else crs_dict

        self.areas = areas

        self.targets = targets
        self.feature_cols = [] if feature_cols is None else feature_cols
        self.stats = [] if stats is None else stats
        # difference between measurement and pointclouds taken
        self.radius = xy_radius

        # different types of targets
        self.reg_targets = [target for target in self.targets if self.targets[target]["task"] == "regression"]
        self.cls_targets = [target for target in self.targets if self.targets[target]["task"] == "classification"]
        self.cls_targets_ = [f"{target}_" for target in self.targets if
                             self.targets[target]["task"] == "classification"]
        self.mol_targets = [target for target in self.targets if self.targets[target]["task"] == "mol"]

        # if not give, calculate on given data
        if feature_scaling_dict is None:
            feature_scaling_dict = {
                area_name:
                    {  # feature: (center, scale)
                        "num_returns": (0., 5.),
                        "return_num": (0., 5.), }
                for area_name in areas
            }
        self.feature_scaling_dict = feature_scaling_dict

        super().__init__(
            root, transform, pre_transform, pre_filter
        )
        # check if all areas are actually processed when using saves
        if self.save_processed:
            for area_name in areas:
                area = areas[area_name]
                labels = area["labels"].query(f"{area['split_col']} == '{self.split}'")
                if len(labels) > 0 and not (Path(self.processed_dir) / self.split / area_name / "done.flag").exists():
                    log.info(f'Resuming processing, since {area_name} is not complete!')
                    self.process()
        else:
            self.process()

        # pre-load into memory if not already done during processing
        if self.in_memory:
            log.info("Pre-loading into memory")
            pbar = tqdm(range(len(self)), total=len(self))
            [self.get(idx) for idx in pbar]


    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder)

    @property
    def raw_file_names(self):
        files = list(chain(*[[area["pt_files"]] for area in self.areas.values()]))
        return files

    @property
    def has_labels(self) -> bool:
        return self.split in ["val", "test"]

    @property
    def processed_file_names(self):
        path = Path(self.processed_dir) / self.split
        files = glob(str(path / f"*/*.pt"))
        return files

    @property
    def num_samples(self):
        n = 0
        if self.in_memory and not self.save_processed:
            n = len(self.memory)
            if n == 0:  # memory not initialized yet
                n = sum([len(area["labels"].query(f"{area['split_col']} == '{self.split}'")) for area in self.areas])
            return n

        for area_name in self.areas:
            area = self.areas[area_name]

            if (Path(self.processed_dir) / self.split / area_name / "done.flag").exists():
                n += len(list((Path(self.processed_dir) / self.split / area_name).glob("*.pt")))
            else:
                n += len(area["labels"].query(f"{area['split_col']} == '{self.split}'"))

        return n

    def process(self):
        file_idx = 0

        for area_name in self.areas:
            flag = (Path(self.processed_dir) / self.split / area_name / "done.flag")
            area = self.areas[area_name]

            log.info(f"### start processing area: '{area_name}'")
            if not flag.exists():

                labels = area["labels"].query(f"{area['split_col']} == '{self.split}'")
                if len(labels) == 0:
                    continue

                if area["type"] == "scene":
                    # can prepare this beforehand
                    pos, features, inner_label_point_idx, label_point_idx, labels = \
                        self.process_scene_area_(area_name, labels)

                ### TODO reintroduce feature scaling
                # if feature in feature_scaling:
                #     center, scale = feature_scaling.get(feature, (0., 1.))
                # else:
                #     # fill with iqr scaling
                #     center = np.median(feat)
                #     scale = (np.quantile(feat, 0.75) - np.quantile(feat, 0.25)) * 1.349
                #     feature_scaling[feature] = (center, scale)
                # features_sample.append((feat - center) / scale)

                log.info("Saving samples and calculating stats")
                if self.save_processed:
                    (Path(self.processed_dir) / self.split).mkdir(exist_ok=True)
                    (Path(self.processed_dir) / self.split / area_name).mkdir(exist_ok=True)
                missing_idx = []
                for idx, index in tqdm(enumerate(labels.index.values)):
                    sample = labels.iloc[idx]
                    file = Path(self.processed_dir) / self.split / area_name / f"{file_idx}.pt"
                    if file.exists():
                        file_idx += 1
                        continue

                    if area["type"] == "object":
                        # only load objects here instead of bulk loading before to avoid memory issues
                        pos, features, crs = read_pt(sample["pt_file"], self.feature_cols, area["delimiter"])

                        if area.get("check_pt_crs", True) and crs is not None and \
                                not pyproj.CRS.is_exact_same(labels.crs, crs):
                            sample = labels.to_crs(crs).iloc[idx]

                        # find points
                        label_centers = [[sample.geometry.x, sample.geometry.y]]
                        tree = KDTree(pos[:, :2])
                        point_idxs = tree.query_radius(label_centers, self.radius)[0]
                        inner_point_idx = tree.query_radius(label_centers, self.radius / 2.)[0]
                        del tree

                    elif area["type"] == "scene":
                        point_idxs = label_point_idx[idx]
                        inner_point_idx = inner_label_point_idx[idx]
                    else:
                        raise NotImplementedError("Only 'scence' and 'object' area types are implemented")

                    data = self.save_data_(
                        area_name, index, sample, pos, features,
                        point_idxs, inner_point_idx
                    )
                    if data is not None:
                        if self.in_memory:
                            self.memory[file_idx] = data
                        if self.save_processed:
                            torch.save(data, file)
                        file_idx += 1
                    else:
                        missing_idx.append(index)
                area["labels"].drop(index=missing_idx, inplace=True)
                if self.save_processed:
                    flag.touch()
            else:
                file_idx += len(list((Path(self.processed_dir) / self.split / area_name).glob("*.pt")))

    def process_scene_area_(self, area_name, labels):
        area = self.areas[area_name]
        pos_tree = self.pos_tree_dict.get(area_name, None)

        if not pos_tree:
            log.info(f"Loading Las files")
            pt = [read_pt(las_file, self.feature_cols, area["delimiter"]) for las_file in area["pt_files"]]

            pos = np.concatenate([p[0] for p in pt], 0)
            if len(self.feature_cols) > 0:
                features = np.concatenate([p[1] for p in pt], 0)
            else:
                features = None

            crs = np.stack([p[2] for p in pt], 0)
            assert np.all(crs[0] == crs_ for crs_ in crs), "pt_files of an area need to be in same crs currently"
            crs = crs[0]

            # fit this into a KDTree
            log.info("Creating KDTree")
            pos_tree = KDTree(pos[:, :2])

            self.pos_dict[area_name] = pos
            self.pos_tree_dict[area_name] = pos_tree
            self.features_dict[area_name] = features
            self.crs_dict[area_name] = crs
        log.info("Querying KDTree")
        # restrict to bounds
        crs = self.crs_dict[area_name]
        if area.get("check_pt_crs", True) and crs is not None and not pyproj.CRS.is_exact_same(labels.crs, crs):
            labels = labels.to_crs(crs)

        label_centers = np.stack([labels.geometry.x, labels.geometry.y], 1)
        radius = self.radius
        label_point_idx = self.pos_tree_dict[area_name].query_radius(label_centers, radius)
        inner_label_point_idx = self.pos_tree_dict[area_name].query_radius(label_centers, radius / 2.)
        return self.pos_dict[area_name], self.features_dict[area_name], inner_label_point_idx, label_point_idx, labels

    @property
    def num_classes(self) -> int:
        if not hasattr(self, "num_classes_"):
            num_reg_classes = 0
            num_mol_classes = 0
            num_cls_classes = []
            if self.targets:
                for target in self.targets:
                    task = self.targets[target]["task"]
                    if task == "classification":
                        num_cls_classes.append(len(self.targets[target]["class_names"]))
                    elif task == "regression":
                        num_reg_classes += 1
                    elif task.lower() == "mol":
                        num_mixtures = self.targets[target].get("num_mixtures", 1)
                        num_mol_classes += num_mixtures * 3

                self.num_reg_classes_ = num_reg_classes
                self.num_mol_classes_ = num_mol_classes
                self.num_cls_classes_ = num_cls_classes

            self.num_classes_ = self.num_reg_classes + self.num_mol_classes + int(np.sum(self.num_cls_classes))

        return self.num_classes_

    @property
    def num_reg_classes(self) -> int:
        if not hasattr(self, "num_reg_classes_"):
            # init by calling num_classes
            _ = self.num_classes

        return self.num_reg_classes_

    @property
    def num_mol_classes(self) -> int:
        if not hasattr(self, "num_mol_classes_"):
            # init by calling num_classes
            _ = self.num_classes

        return self.num_mol_classes_

    @property
    def num_cls_classes(self) -> []:
        if not hasattr(self, "num_cls_classes_"):
            # init by calling num_classes
            _ = self.num_classes

        return self.num_cls_classes_

    def len(self):
        return self.num_samples

    @staticmethod
    def get_local_stats(points, postfix=""):
        stats = {}
        z = points[:, 2]

        z_stats = {
            "h_mean": np.mean,
            "h_std": np.std,
            "h_coov": scstats.variation,
            "h_kur": scstats.kurtosis,
            "h_skew": scstats.skew,
        }

        quantiles = [5, 10, 25, 50, 75, 90, 95, 99]
        z_stats.update({f"h_q{i}": partial(np.quantile, q=i / 100) for i in quantiles})

        def density_q(z, q):
            # the proportion of points above the height percentiles
            quant = np.quantile(z, q=q)
            return len(z[z > quant]) / len(z)

        z_stats.update({f"d_q{i}": partial(density_q, q=i / 100) for i in quantiles})

        tree = KDTree(points)
        # create 1m grid spanning extend
        xx = np.arange(points[:, 0].min(), points[:, 0].max(), 1)
        yy = np.arange(points[:, 1].min(), points[:, 1].max(), 1)
        zz = np.arange(points[:, 2].min(), points[:, 2].max(), 1)
        grid = [[x, y, z] for x, y, z in product(xx, yy, zz)]
        # get highest density in grid
        if len(grid) > len(points):  # use points directly if only few points present
            grid = points
        density = tree.kernel_density(grid, 1, kernel="gaussian").max()
        stats["kde_h1"] = density

        for key in z_stats.keys():
            try:
                value = z_stats[key](z)
            except IndexError:
                # return -1 if not enough values in quantiles
                value = -1

            stats[key + postfix] = value

        return stats

    def get(self, idx):
        if self.in_memory:
            if idx in self.memory.keys():
                data = self.memory[idx].clone()
            else:
                data = torch.load(self.processed_file_names[idx])
                self.memory[idx] = data.clone()
        else:
            data = torch.load(self.processed_file_names[idx])

        del data.local_stats_keys
        data["is_double"] = self.prev_idx == idx
        self.prev_idx = idx

        return data

    def save_data_(self, area_name: str, idx, sample, pos_: np.array, features_: np.array,
                   point_idxs: np.array, inner_point_idxs: np.array):

        if len(point_idxs) < self.min_pts_outer:
            log.warning(f"only {len(point_idxs)} in total, skipping")
            return None
        elif len(inner_point_idxs) < self.min_pts_inner:
            log.warning(f"only {len(inner_point_idxs)} in inner circle, skipping")
            return None

        # only coordinates for now
        x = pos_[point_idxs]
        inner_x = pos_[inner_point_idxs]

        if features_ is not None:
            features = features_[point_idxs]
        else:
            features = None

        # normalize
        inner_x, x = self.center_pos(x, inner_x, sample)

        # get local and df stats
        local_stats, local_stats_keys, stats = self.get_stats(x, inner_x, sample)

        # target
        if self.targets:
            y_reg = sample[self.reg_targets]
            y_reg_mask = ~y_reg.isna()
            y_mol = sample[self.mol_targets]
            y_mol_mask = ~y_mol.isna()
            y_cls = sample[self.cls_targets_]
            y_cls_mask = ~y_cls.isna()

        else:
            y_reg = y_reg_mask = y_mol = y_mol_mask = y_cls = y_cls_mask = []

        data = self.covert_to_data_(
            x, y_reg, y_reg_mask, y_mol, y_mol_mask, y_cls, y_cls_mask,
            features, idx, area_name, local_stats, local_stats_keys, stats
        )

        return data

    def covert_to_data_(
            self, x, y_reg, y_reg_mask, y_mol, y_mol_mask, y_cls, y_cls_mask, features, idx, area_name, local_stats,
            local_stats_keys, stats
    ):
        x = torch.tensor(x, dtype=torch.float32)
        y_reg = torch.tensor(y_reg, dtype=torch.float32)
        y_reg_mask = torch.tensor(y_reg_mask, dtype=torch.bool)
        y_mol = torch.tensor(y_mol, dtype=torch.float32)
        y_mol_mask = torch.tensor(y_mol_mask, dtype=torch.bool)
        y_cls[~y_cls_mask] = - 1
        y_cls = torch.tensor(y_cls, dtype=torch.long)
        y_cls_mask = torch.tensor(y_cls_mask, dtype=torch.bool)
        features = features if features is None else torch.tensor(features, dtype=torch.float32)
        stats = torch.tensor(stats, dtype=torch.float32)
        local_stats = torch.tensor(local_stats, dtype=torch.float32)
        data = Data(
            x=features,
            y_reg=y_reg, y_reg_mask=y_reg_mask,
            y_mol=y_mol, y_mol_mask=y_mol_mask,
            y_cls=y_cls, y_cls_mask=y_cls_mask,
            pos=x, stats=stats, label_idx=[idx], area_name=area_name,
            local_stats=local_stats, local_stats_keys=local_stats_keys
        )

        # apply pre_transform
        if self.pre_transform is not None:
            data = self.pre_transform(data)
            if data.pos.shape[0] == 0:
                log.warning(f"Pre transform reduced sample to 0 points, skipping")
                return None

        return data

    def get_stats(self, x, inner_x, df):
        # local stats
        if self.save_local_stats:
            local_stats = self.get_local_stats(x)
            local_stats.update(self.get_local_stats(inner_x, "_inner"))
            local_stats_keys = list(local_stats.keys())
            local_stats = list(local_stats.values())
        else:
            local_stats = local_stats_keys = []
        # global stats
        stats = df[self.stats]
        return local_stats, local_stats_keys, stats

    def center_pos(self, x, inner_x, df):
        x_center = np.amin(x, axis=0, keepdims=True)
        x_center[:, 0] = df.geometry.x
        x_center[:, 1] = df.geometry.y
        x -= x_center
        inner_x -= x_center
        return inner_x, x


class LasDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.dataset_opt = dataset_opt
        self.targets = dataset_opt.get("targets", None)
        self.target_keys = list(self.targets.keys()) if self.targets is not None else None
        self.features = dataset_opt.features
        self.stats = dataset_opt.stats
        self.xy_radius = dataset_opt.xy_radius
        self.x_scale = dataset_opt.x_scale
        self.y_scale = dataset_opt.y_scale
        self.z_scale = dataset_opt.z_scale
        self.transform_type = dataset_opt.transform_type
        self.double_batch = dataset_opt.get(self.transform_type).get("double_batch", False)
        self.log_train_metrics = dataset_opt.get("log_train_metrics", True)

        self.reg_targets = [target for target in self.targets if self.targets[target]["task"] == "regression"]
        self.reg_targets_idx = [self.targets[target]["task"] == "regression" for target in self.targets]
        self.cls_targets = [target for target in self.targets if self.targets[target]["task"] == "classification"]
        self.cls_targets_idx = [self.targets[target]["task"] == "classification" for target in self.targets]
        self.cls_targets_ = [f"{target}_" for target in self.cls_targets]
        self.mol_targets = [target for target in self.targets if self.targets[target]["task"] == "mol"]
        self.mol_targets_idx = [self.targets[target]["task"] == "mol" for target in self.targets]

        self.min_pts_outer = dataset_opt.get("min_pts_outer", 500)
        self.min_pts_inner = dataset_opt.get("min_pts_inner", 250)

        in_memory = dataset_opt.get("in_memory", False)
        save_processed = dataset_opt.get("save_processed", True)
        save_local_stats = dataset_opt.get("save_local_stats", True)
        train_subset = dataset_opt.get("train_subset", False)

        processed_folder = dataset_opt.get("processed_folder", "processed")

        areas_file = (self._data_path / (Path(processed_folder)) / "areas.pt")
        self.areas: dict = OrderedDict(OmegaConf.to_container(dataset_opt.areas))
        if areas_file.exists():
            self.areas.update(torch.load(areas_file))
        self.process_area_labels(dataset_opt)
        train_set_avail = any(
            [len(area["labels"].query(f"{area['split_col']} == 'train'")) > 0 for area in self.areas.values()])
        val_set_avail = any(
            [len(area["labels"].query(f"{area['split_col']} == 'val'")) > 0 for area in self.areas.values()])
        test_set_avail = any(
            [len(area["labels"].query(f"{area['split_col']} == 'test'")) > 0 for area in self.areas.values()])

        if save_processed:
            (self._data_path / (Path(processed_folder))).mkdir(exist_ok=True)

        feature_scaling_file = self._data_path / (Path(processed_folder) / "features_scaling.pt")
        feature_scaling_dict = torch.load(feature_scaling_file) if feature_scaling_file.exists() else None

        assert train_set_avail or val_set_avail or test_set_avail, "Apparently no data available"

        pos_dict = {}
        pos_tree_dict = {}
        features_dict = {}
        crs_dict = {}
        if train_set_avail:
            if train_subset:
                train_subset_remove = 1 - train_subset
                for area in self.areas.values():
                    idx = area["labels"].query(f"{area['split_col']} == 'train'").index
                    idx = np.random.choice(idx, int(len(idx) * train_subset_remove), replace=False)
                    area["labels"].drop(index=idx, inplace=True)

            log.info("Init train dataset")
            self.train_dataset = Las(
                self._data_path, areas=self.areas, split="train",
                targets=self.targets, feature_cols=self.features, feature_scaling_dict=feature_scaling_dict,
                stats=dataset_opt.stats, transform=self.train_transform, pre_transform=self.pre_transform,
                save_processed=save_processed, processed_folder=processed_folder, in_memory=in_memory,
                xy_radius=self.xy_radius, save_local_stats=save_local_stats,
                min_pts_outer=self.min_pts_outer, min_pts_inner=self.min_pts_inner
            )
            if not feature_scaling_file.exists():
                feature_scaling_dict = self.train_dataset.feature_scaling_dict
                torch.save(feature_scaling_dict, feature_scaling_file)

            pos_dict.update(self.train_dataset.pos_dict)
            pos_tree_dict.update(self.train_dataset.pos_tree_dict)
            features_dict.update(self.train_dataset.features_dict)
            crs_dict.update(self.train_dataset.crs_dict)

        if val_set_avail:
            log.info("Init val dataset")
            self.val_dataset = Las(
                self._data_path, areas=self.areas, split="val",
                targets=self.targets, feature_cols=self.features, feature_scaling_dict=feature_scaling_dict,
                stats=dataset_opt.stats, transform=self.val_transform, pre_transform=self.pre_transform,
                save_processed=save_processed, processed_folder=processed_folder, in_memory=in_memory,
                xy_radius=self.xy_radius, save_local_stats=save_local_stats,
                min_pts_outer=self.min_pts_outer, min_pts_inner=self.min_pts_inner,
                pos_dict=pos_dict, features_dict=features_dict,
                pos_tree_dict=pos_tree_dict, crs_dict=crs_dict
            )

            pos_dict.update(self.val_dataset.pos_dict)
            pos_tree_dict.update(self.val_dataset.pos_tree_dict)
            features_dict.update(self.val_dataset.features_dict)
            crs_dict.update(self.val_dataset.crs_dict)

        if test_set_avail:
            log.info("Init test dataset")
            self.test_dataset = Las(
                self._data_path, areas=self.areas, split="test",
                targets=self.targets, feature_cols=self.features, feature_scaling_dict=feature_scaling_dict,
                stats=dataset_opt.stats, transform=self.test_transform, pre_transform=self.pre_transform,
                save_processed=save_processed, processed_folder=processed_folder, in_memory=in_memory,
                xy_radius=self.xy_radius, save_local_stats=save_local_stats,
                min_pts_outer=self.min_pts_outer, min_pts_inner=self.min_pts_inner,
                pos_dict=pos_dict, features_dict=features_dict,
                pos_tree_dict=pos_tree_dict, crs_dict=crs_dict
            )

        del pos_dict, pos_tree_dict, features_dict, crs_dict

        # save areas in preprocessed file
        if save_processed:
            torch.save(self.areas, areas_file)

        self.set_label_stats_(save_processed)

        self.has_reg_targets = len(self.reg_targets) > 0
        self.has_mol_targets = len(self.mol_targets) > 0
        self.has_cls_targets = len(self.cls_targets) > 0

    def process_area_labels(self, dataset_opt):
        for area_name in self.areas:
            area = self.areas[area_name]

            # assume that if the labels are set, the area was already processed
            if area.get("labels", None) is not None:
                continue

            # set some standard params
            area["delimiter"] = area.get("delimiter", dataset_opt.get("delimiter", ","))

            # processing file lists
            pt_files = area["pt_files"]
            if isinstance(pt_files, (str, Path)):
                pt_files = glob(str(Path(self._data_path) / "raw" / pt_files))
            elif isinstance(pt_files, list):
                # iterating to list of files
                unpacked_files = []
                for f in pt_files:
                    unpacked_files.extend(glob(str(Path(self._data_path) / "raw" / f)))
                pt_files = unpacked_files
            else:
                raise Exception("pt_files need to be a str or a list of str (can use * expression)")

            labels = self.process_label_files_(area, area_name)

            labels.geometry = labels.centroid

            if area["type"] == "object":
                # check if each label has a pt_file
                def find_pt_file(id):
                    for ptf in pt_files:
                        # return first occurrence
                        if id in ptf:
                            return ptf
                    return "None"

                labels["pt_file"] = labels[area["pt_identifier"]].apply(find_pt_file)

                # removing sample without pt_file
                n_samples = len(labels)
                labels.query("pt_file != 'None'", inplace=True)
                if len(labels) != n_samples:
                    log.warning(f"{n_samples - len(labels)} removed due to missing pt_file")

                pt_files = labels["pt_file"].values.tolist()

            area["pt_files"] = pt_files

            split_col = area.get("split_col", dataset_opt.get("split_col", "split"))
            area["split_col"] = split_col
            # create split if fully labeled data available
            if split_col not in labels.columns:
                targets_must_be_present = np.array(area.get("targets_must_be_present", [True] * len(self.target_keys)))
                lb = labels[np.array(self.target_keys)[targets_must_be_present]]

                val_ratio = area.get("val_ratio", .1)
                test_ratio = area.get("test_ratio", .1)

                # if no targets are fully available, only use this area for training
                if (lb.shape[1] > 0 and lb.isna().all().all()) or val_ratio == test_ratio == 0.0:
                    labels.loc[:, split_col] = "train"
                else:
                    # no split available, create own
                    # only select those that have labels others are for training
                    if any(targets_must_be_present):
                        partly_missing = lb.isna().all(axis=1)
                        lables_partly_missing = labels[partly_missing]
                        lables_partly_missing[split_col] = "train"

                        lables_full = labels[~partly_missing]
                    else:
                        lables_partly_missing = pd.DataFrame()
                        lables_full = labels
                    index = lables_full.index.values

                    rs = np.random.RandomState(42)

                    rs.shuffle(index)

                    train_end = int(len(index) * (1 - (val_ratio + test_ratio)))
                    val_end = int(len(index) * (1 - test_ratio))
                    train_idx = index[:train_end]
                    val_idx = index[train_end:val_end]
                    test_idx = index[val_end:]

                    lables_full.loc[train_idx, split_col] = "train"
                    if val_ratio != 0 and len(val_idx) > 0:
                        lables_full.loc[val_idx, split_col] = "val"
                    if test_ratio != 0 and len(test_idx) > 0:
                        lables_full.loc[test_idx, split_col] = "test"

                    labels = pd.concat([lables_partly_missing, lables_full])

                if len(labels.query(f"['val', 'test'] in {split_col}")) == 0:
                    log.warning(f"neither val nor test set present for {area_name}")

            area["labels"] = labels

    def process_label_files_(self, area: dict, area_name: str):
        label_files = area["label_files"]
        # ensure labels file follows schemata:
        #    [file_1, ..., file_n]
        if isinstance(label_files, (str, Path)):
            label_files = [label_files]

        assert len(label_files) > 0, f"no labels given, check area {area_name}"

        labels = None
        for lf in label_files:
            lb = gpd.read_file(Path(self._data_path) / "raw" / lf)

            # put dummy point if no position exists (usually true for csv data)
            lb.geometry = lb.geometry.apply(lambda g: Point(0, 0) if g is None else g)

            alias_targets = area.get("alias_targets", self.targets)
            assert len(alias_targets) == len(self.targets), f"given target aliases for '{area_name}' have " \
                                                            f"different lengths: {alias_targets} vs {self.targets}"

            target_metric_factor = area.get("target_metric_factor", None)

            # add targets if present else set to nan
            for ori_target, alias_target in zip(self.targets, alias_targets):
                task = self.targets[ori_target]["task"]
                if alias_target in lb:
                    lb[ori_target] = lb[alias_target]
                    # assumes that classification targets will be not necessarily be numbers, but everything else is
                    if task in ["regression", "mol"]:
                        lb[ori_target] = pd.to_numeric(lb[ori_target], errors="coerce")
                        if target_metric_factor is not None:
                            lb[ori_target] *= target_metric_factor.get(ori_target, 1.0)
                else:
                    lb[ori_target] = np.nan

                if task == "classification":
                    # also save numerical values according to given classes
                    lb[f"{ori_target}_"] = lb[ori_target].map(
                        self.targets[ori_target]["class_mapping"]
                    ).astype(float)

            # crs comparison
            if labels is None:
                labels = lb
                crs = lb.crs
            else:
                if crs != lb.crs:
                    Warning("CRS of label files do not match, have to convert")
                    lb = lb.to_crs(crs)
                labels = pd.concat([labels, lb])

        # indicate fully/partly missing targets in label sample
        n_labels = len(labels)
        nans_allowed = area.get("nans_allowed", True)
        fully_missing = labels[self.targets].isna().all(axis=1).sum()
        partly_missing = labels[self.targets].isna().any(axis=1).sum()
        partly_missing = abs(partly_missing - fully_missing)
        if fully_missing > 0:
            log.info(f"{fully_missing} of {n_labels} labels fully missing in {area_name}")
            if fully_missing == n_labels:
                area["has_labels"] = False
        if partly_missing > 0:
            log.info(f"{partly_missing} of {n_labels} labels partly missing in {area_name}")
            if fully_missing + partly_missing == n_labels and not nans_allowed:
                area["has_labels"] = False

        if not nans_allowed:
            labels.dropna(axis=0, how="any", subset=self.targets, inplace=True)
            log.info(
                f"Removing all missing or partly missing samples as indicated by 'nans_allowed' in {area_name}"
            )

        # apply filter query
        query = area.get("label_query", None)
        if query is not None:
            labels.query(query, inplace=True)
            if n_labels > len(labels):
                log.warning(f"({n_labels - len(labels)} sample were "
                            f"filtered out according to: {query})")

        labels.set_index(np.arange(len(labels)), inplace=True)
        return labels

    def set_label_stats_(self, save_processed: bool):
        processed_dir = Path(os.path.join(self._data_path, self.dataset_opt.processed_folder))
        if save_processed:
            processed_dir.mkdir(exist_ok=True)
        means_file = processed_dir / "mean_targets.pt"
        std_file = processed_dir / "std_targets.pt"
        min_file = processed_dir / "min_targets.pt"
        max_file = processed_dir / "max_targets.pt"
        corr_file = processed_dir / "corr_targets.pt"

        self.mean_targets_ = torch.load(means_file) if means_file.exists() else \
            self.get_stat_targets_(np.nanmean, means_file if save_processed else None)
        self.std_targets_ = torch.load(std_file) if std_file.exists() else \
            self.get_stat_targets_(np.nanstd, std_file if save_processed else None)
        self.min_targets_ = torch.load(min_file) if min_file.exists() else \
            self.get_stat_targets_(np.nanmin, min_file if save_processed else None)
        self.max_targets_ = torch.load(max_file) if max_file.exists() else \
            self.get_stat_targets_(np.nanmax, max_file if save_processed else None)

        self.corr_targets_ = torch.load(corr_file) if corr_file.exists() else \
            self.get_corr_targets_(corr_file if save_processed else None)

    def create_dataloaders(
            self,
            model: model_interface.DatasetInterface,
            batch_size: int,
            shuffle: bool,
            drop_last: bool,
            num_workers: int,
            precompute_multi_scale: bool,
    ):
        if self.train_dataset and shuffle:
            self.train_sampler = RandomSampler(self.train_dataset, batch_size, self.double_batch)
            if drop_last is False:
                log.warning("Cannot disable 'drop_last' with RandomSampler.")
        super().create_dataloaders(model, batch_size, shuffle, drop_last, num_workers, precompute_multi_scale)

    def get_std_targets(self):
        return self.std_targets_

    def get_mean_targets(self):
        return self.mean_targets_

    def get_min_targets(self):
        return self.min_targets_

    def get_max_targets(self):
        return self.max_targets_

    def get_stat_targets_(self, stat_fn, file_name: (str, Path) = None):
        dict = OrderedDict()
        targets = [f"{target}_" if self.targets[target]["task"] == "classification" else target for target in
                   self.targets]

        dict["total"] = {}
        if self.train_dataset is not None:
            dict["total"].update({"train": [], })
        if self.val_dataset is not None:
            dict["total"].update({"val": [], })
        if self.test_dataset is not None:
            dict["total"].update({"test": [], })

        for area_name in self.areas:
            # TODO also uses labels that were not used due to too few points
            sc = self.areas[area_name]["split_col"]
            labels = self.areas[area_name]["labels"]
            area_dict = {}
            if self.train_dataset is not None and labels.query(f"{sc} == 'val'").shape[0] > 1:
                values = labels.query(f"{sc} == 'train'")[targets].values
                area_dict.update({"train": stat_fn(values, 0), })
                dict["total"]["train"].append(values)
            if self.val_dataset is not None and labels.query(f"{sc} == 'val'").shape[0] > 1:
                values = labels.query(f"{sc} == 'val'")[targets].values
                area_dict.update({"val": stat_fn(values, 0), })
                dict["total"]["val"].append(values)
            if self.test_dataset is not None and labels.query(f"{sc} == 'test'").shape[0] > 1:
                values = labels.query(f"{sc} == 'test'")[targets].values
                area_dict.update({"test": stat_fn(values, 0), })
                dict["total"]["test"].append(values)

            if len(area_dict) > 0:
                dict[area_name] = area_dict

        if self.train_dataset is not None:
            dict["total"]["train"] = stat_fn(np.concatenate(dict["total"]["train"], 0), 0)
        if self.val_dataset is not None:
            dict["total"]["val"] = stat_fn(np.concatenate(dict["total"]["val"], 0), 0)
        if self.test_dataset is not None:
            dict["total"]["test"] = stat_fn(np.concatenate(dict["total"]["test"], 0), 0)

        if file_name is not None:
            torch.save(dict, file_name)

        return dict

    def get_corr_targets(self):
        return self.corr_targets_

    def get_corr_targets_(self, file_name: (str, Path) = None):
        dict = OrderedDict()
        targets = [f"{target}_" if self.targets[target]["task"] == "classification" else target for target in
                   self.targets]

        for area_name in self.areas:
            sc = self.areas[area_name]["split_col"]
            labels = self.areas[area_name]["labels"]
            area_dict = {}
            if self.train_dataset is not None and labels.query(f"{sc} == 'train'").shape[0] > 1:
                area_dict.update({"train": labels.query(f"{sc} == 'train'")[targets].corr().values, })
            if self.val_dataset is not None and labels.query(f"{sc} == 'val'").shape[0] > 1:
                area_dict.update(
                    {"val": labels.query(f"{sc} == 'val'")[targets].corr().values, }
                )
            if self.test_dataset is not None and labels.query(f"{sc} == 'test'").shape[0] > 1:
                area_dict.update(
                    {"test": labels.query(f"{sc} == 'test'")[targets].corr().values, }
                )

            if len(area_dict) > 0:
                dict[area_name] = area_dict

        if file_name is not None:
            torch.save(dict, file_name)
        return dict

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return InstanceTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log,
                               log_train_metrics=self.log_train_metrics)

    @property  # type: ignore
    @save_used_properties
    def num_reg_classes(self) -> int:
        if self.train_dataset:
            return self.train_dataset.num_reg_classes
        elif self.test_dataset is not None:
            if isinstance(self.test_dataset, list):
                return self.test_dataset[0].num_reg_classes
            else:
                return self.test_dataset.num_reg_classes
        elif self.val_dataset is not None:
            return self.val_dataset.num_reg_classes
        else:
            raise NotImplementedError()

    @property  # type: ignore
    @save_used_properties
    def num_mol_classes(self) -> int:
        if self.train_dataset:
            return self.train_dataset.num_mol_classes
        elif self.test_dataset is not None:
            if isinstance(self.test_dataset, list):
                return self.test_dataset[0].num_mol_classes
            else:
                return self.test_dataset.num_mol_classes
        elif self.val_dataset is not None:
            return self.val_dataset.num_mol_classes
        else:
            raise NotImplementedError()

    @property  # type: ignore
    @save_used_properties
    def num_cls_classes(self) -> int:
        if self.train_dataset:
            return self.train_dataset.num_cls_classes
        elif self.test_dataset is not None:
            if isinstance(self.test_dataset, list):
                return self.test_dataset[0].num_cls_classes
            else:
                return self.test_dataset.num_cls_classes
        elif self.val_dataset is not None:
            return self.val_dataset.num_cls_classes
        else:
            raise NotImplementedError()


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): number of samples in a mini-batch
        double_batch (bool): if each sample should in a batch should be returned twice (e.g., for self-supervision)
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, data_source: Sized, batch_size: int, double_batch: bool, generator=None) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = generator
        self._num_samples = None
        self.batch_size = batch_size
        self.double_batch = double_batch

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        iterator = torch.randperm(self.num_samples, generator=generator).tolist()
        if self.double_batch:
            iterator = np.array([[k, k] for k in iterator]).flatten().tolist()
        iterator = iterator[:(self.num_samples // self.batch_size) * self.batch_size]

        yield from iterator

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self):
        return "{}(batch_size={},double_batch={},generator={})".format(
            self.__class__.__name__, self.batch_size, self.double_batch, self.generator,
        )
