#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc GatherScatterKernel.cu -o build/GatherScatterKernel.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc ComputePermutationInfo.cu -o build/ComputePermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc ComputeRepermutationInfo.cu -o build/ComputeRepermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc ComputeVoxelIdx.cu -o build/ComputeVoxelIdx.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc SearchNeighborhood.cu -o build/SearchNeighborhood.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

cu_lib="build/GatherScatterKernel.cu.o
        build/ComputePermutationInfo.cu.o
        build/ComputeRepermutationInfo.cu.o
        build/ComputeVoxelIdx.cu.o
        build/SearchNeighborhood.cu.o
        "

cc_file="SumGather.cc
         MaxGather.cc
         MaxScatter.cc
         ComputePermutationInfo.cc
         ComputePermutationInfo.cpp
         ComputeRepermutationInfo.cc
         ComputeVoxelIdx.cc
         SearchNeighborhood.cc
         "

g++ -std=c++11 -shared ${cu_lib} ${cc_file} -o build/PCSOps.so \
         -fPIC -I$TF_INC -I/home/liuyuan/lib/include -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -L/usr/local/cuda/lib64/libcudart.so \
         -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -pthread