#!/bin/bash

# Compile cpp subsampling
cd torch_points3d/modules/KPConv/cpp_wrappers/cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ../../../..