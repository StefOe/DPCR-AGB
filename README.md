# ReadMe

This is the repository for the *Remote Sensing of Environment* article: *Deep Point Cloud Regression for
Above-Ground Forest Biomass Estimation from Airborne LiDAR*.

We include **code**, **evaluation scripts**, **model weights** (soon), and the **dataset** (partly, soon all).

Regarding the code:
We forked the [torch-points3d](https://github.com/nicolas-chaulet/torch-points3d) framework and added support for
regression tasks including datasets, tracking, and models on our own. In the process, we also simplified the usage of
package.

In addition, we also included our code to load the trained linear regression and random forest in
the `pointcloud_stats_method` folder. Just run the notebook `learn_with_stats.ipynb`.

Finally, the results/plots for each method can be seen in the `eval_scripts` folder within
the `eval_deep_learning_v2.ipynb`. The results for the network size experiment are in `eval_deep_learning_v2_size.ipynb`.

**results on the test set:**

| target          | model    | treeadd |    $R^2$ |       |     RMSE |         |      MAPE |           | mean bias |          |
|:----------------|:---------|:--------|---------:|------:|---------:|--------:|----------:|----------:|----------:|---------:|
|                 |          |         | *median* | *max* | *median* |   *min* |  *median* |     *min* |  *median* |    *min* |
| **biomass**     | KPConv   | False   |    0.800 | 0.815 |   45.264 |  43.540 |   396.685 |   272.288 |     0.460 |    0.389 |
|                 |          | True    |    0.780 | 0.803 |   47.526 |  44.975 |   467.581 |   246.927 |     3.660 |   -0.707 |
|                 | MSENet14 | False   |    0.825 | 0.829 |   42.373 |  41.806 |   299.497 |   192.777 |     0.666 |   -0.291 |
|                 |          | True    |    0.823 | 0.829 |   42.596 |  41.851 |   271.716 |   131.120 |     0.313 |    0.122 |
|                 | MSENet50 | False   |    0.827 | 0.835 |   42.140 |  41.083 |   469.104 |   174.245 |     0.837 |   -0.114 |
|                 |          | True    |    0.824 | 0.837 |   42.481 |  40.909 |   339.700 |   119.264 |     0.889 |    0.596 |
|                 | PointNet | False   |    0.770 | 0.772 |   48.565 |  48.288 |   889.293 |   625.091 |     0.539 |    0.119 |
|                 |          | True    |    0.766 | 0.768 |   48.932 |  48.753 |   896.835 |   622.713 |     2.464 |    1.774 |
|                 | RF       | False   |    0.754 | 0.754 |   50.188 |  50.158 |   625.439 |   616.635 |     1.470 |    1.459 |
|                 |          | True    |    0.151 | 0.157 |   93.238 |  92.930 |  7644.787 |  7423.094 |    47.625 |  -47.521 |
|                 | power    | False   |    0.761 | 0.761 |   49.509 |  49.509 |   365.606 |   365.606 |     2.027 |    2.027 |
|                 |          | True    |    0.034 | 0.034 |   99.478 |  99.478 |  7604.844 |  7604.844 |    57.525 |  -57.525 |
|                 | linear   | False   |    0.762 | 0.762 |   49.420 |  49.420 |   425.605 |   425.605 |     1.894 |    1.894 |
|                 |          | True    |    0.195 | 0.195 |   90.801 |  90.801 | 11448.501 | 11448.501 |    39.149 |  -39.149 |
| **wood volume** | KPConv   | False   |    0.799 | 0.805 |   85.434 |  84.255 |   103.866 |    85.633 |     0.377 |    0.285 |
|                 |          | True    |    0.778 | 0.792 |   89.808 |  87.002 |   126.543 |    85.812 |     7.885 |   -1.012 |
|                 | MSENet14 | False   |    0.823 | 0.826 |   80.309 |  79.631 |    99.105 |    72.597 |     0.515 |    0.389 |
|                 |          | True    |    0.821 | 0.825 |   80.750 |  79.716 |    84.473 |    70.097 |     2.577 |    1.829 |
|                 | MSENet50 | False   |    0.824 | 0.831 |   79.986 |  78.344 |   131.525 |    72.381 |     0.169 |    0.123 |
|                 |          | True    |    0.822 | 0.832 |   80.571 |  78.177 |   115.634 |    78.422 |     3.572 |    2.646 |
|                 | PointNet | False   |    0.777 | 0.781 |   90.183 |  89.198 |   205.366 |   162.049 |     1.991 |    1.369 |
|                 |          | True    |    0.773 | 0.776 |   90.844 |  90.220 |   236.383 |   174.903 |     5.708 |    4.578 |
|                 | RF       | False   |    0.757 | 0.757 |   94.091 |  94.070 |   223.652 |   222.600 |     3.979 |    3.955 |
|                 |          | True    |    0.192 | 0.197 |  171.475 | 170.930 |  1683.778 |  1676.524 |    85.629 |  -85.465 |
|                 | power    | False   |    0.763 | 0.763 |   92.819 |  92.819 |   223.654 |   223.654 |     4.497 |    4.497 |
|                 |          | True    |    0.120 | 0.120 |  178.973 | 178.973 |  1793.822 |  1793.822 |   101.104 | -101.104 |
|                 | linear   | False   |    0.766 | 0.766 |   92.292 |  92.292 |   171.483 |   171.483 |     4.602 |    4.602 |
|                 |          | True    |    0.243 | 0.243 |  166.034 | 166.034 |  1747.807 |  1747.807 |    72.340 |  -72.340 |

# Install torch-points3d

We setup our environment in the following way (conda is already installed):

1. go to `pointcloud-biomass-estimator/torch-points3d`
2. Make sure to install cuda 11.8 (don't forget to deselect the driver install if your drivers are current)

```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

3. after installing close and reopen the terminal to check if the PATH is set correctly with `echo $PATH`. It should
   **not** have `/usr/local/cuda-10.2` but should have something like `/usr/local/cuda-11.8` in there

5. install mamba (optional but highly recommended)

```
conda install mamba -c conda-forge
```

3. create conda environment:

```
mamba env create -f env.yml
```

or for cpu-version:

```
mamba env create -f env_cpu.yml
```

4. activate environment:

```
mamba activate pts
```

5. install missing pip packages for Minkowski networks

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings blas_include_dirs=${CONDA_PREFIX}/include blas=openblas

```

or for cpu-version:

```
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings blas=openblas

```

5. compile KPConv scripts

```
sh compile_wrappers.sh
```

# Training for Regression

run from within the torch-points3d folder.

*MSENet50:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=SENet50 data=instance/NFI/reg data.transform_type=sparse_xy training=nfi/minkowski lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*MSENet14:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=SENet14 data=instance/NFI/reg data.transform_type=sparse_xy training=nfi/minkowski lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*KPConv:*

```
python -u train.py task=instance models=instance/kpconv model_name=KPConv data=instance/NFI/reg training=nfi/kpconv data.transform_type=xy lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

*PointNet:*

```
python -u train.py task=instance models=instance/minkowski_baseline model_name=MPointNet data=instance/NFI/reg training=nfi/pointnet data.transform_type=sparse_xy lr_scheduler=cosineawr update_lr_scheduler_on=on_num_batch
```

# Calibration batch normalization

to calibrate the trained models batch norm statistics. Note that the checkpoint directory has to be an absolute path,
e.g.: `checkpoint_dir=/home/user/torch-points3d/weights/SENet50/0`

for Minkowski or Pointnet (`model_name=SENet50`, `model_name=SENet14`, or `model_name=MPointNet`):

```
python calibrate_bn.py model_name=${model_name} checkpoint_dir=${checkpoint_dir} data=instance/NFI/reg num_workers=4 task=instance weight_name="total_BMag_ha_rmse" batch_size=64 num_workers=4 data.transform_type=sparse_xy epochs=20
```

for KPConv:

```
python calibrate_bn.py model_name=KPConv checkpoint_dir=${checkpoint_dir} data=instance/NFI/reg num_workers=4 task=instance weight_name="total_BMag_ha_rmse" batch_size=64 num_workers=4 data.transform_type=xy epochs=20
```

# Evaluating our models

run from within the torch-points3d folder. Note that the checkpoint directory has to be an absolute path,
e.g.: `PATHTOFRAMEWORK=/home/user/torch-points3d`
Also, there are 5 weights for each model (from different trials): `TRIAL=1`

*MSENet50:*

```
python eval.py model_name=SENet50 checkpoint_dir=${PATHTOFRAMEWORK}/weights/SENet50/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/msenet50/eval`.

*MSENet14:*

```
python eval.py model_name=SENet14 checkpoint_dir=${PATHTOFRAMEWORK}/weights/SENet14/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/msenet14/eval`.

*KPConv:*

```
python eval.py model_name=KPConv checkpoint_dir=${PATHTOFRAMEWORK}/weights/KPConv/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["val","test"] data.transform_type=xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/kpconv/eval`.

*PointNet:*

```
python eval.py model_name=MPointNet checkpoint_dir=${PATHTOFRAMEWORK}/weights/PointNet/${TRIAL}/ weight_name="latest" batch_size=32 num_workers=4 eval_stages=["val","test"] data.transform_type=sparse_xy_eval data=instance/NFI/reg task=instance
```

the save folder location is `weights/pointnet/eval`.

# Using tree-adding augmentations during test

same as before, but the transform type changes to use tree augmentations, e.g.:

```
python eval.py model_name=MPointNet checkpoint_dir=${PATHTOFRAMEWORK}/weights/pointnet/ weight_name="total_rmse" batch_size=32 num_workers=4 eval_stages=["val","test"] data.transform_type=sparse_xy_eval_treeadd
```
