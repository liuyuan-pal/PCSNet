//
// Created by pal on 18-3-1.
//

#ifndef CUDACOMMON_H
#define CUDACOMMON_H

#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline int infTwoExp(int val)
{
    int inf=1;
    while(val>inf) inf<<=1;
    return inf;
}


inline void getGPULayout(
        int dim0,int dim1,int dim2,
        int& bdim0,int& bdim1,int& bdim2,
        int& tdim0,int& tdim1,int& tdim2
)
{
    tdim2=64;
    if(dim2<tdim2) tdim2=infTwoExp(dim2);
    bdim2=dim2/tdim2;
    if(dim2%tdim2>0) bdim2++;

    tdim1=1024/(tdim2);
    if(dim1<tdim1) tdim1=infTwoExp(dim1);
    bdim1=dim1/tdim1;
    if(dim1%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(dim0<tdim0) tdim0=infTwoExp(dim0);
    bdim0=dim0/tdim0;
    if(dim0%tdim0>0) bdim0++;
}

#endif //CUDACOMMON_H
