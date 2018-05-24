#include "CudaCommon.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

__global__
void countNeighborNumKernel(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int count=0;
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
            count++;
    }
    lens[pi]=count;
}

__global__
void countNeighborNumRangeKernel(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int count=0;
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        float sq_dist=(tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz);
        if(sq_dist<squared_max_nn_size&&sq_dist>squared_min_nn_size)
            count++;
    }
    lens[pi]=count;
}

__global__
void computeNeighborIdxsKernel(
        float * xyzs,               // [pn,3]
        int *begs,                  // [pn]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int* cur_idxs=&idxs[begs[pi]];
    int* cur_cens=&cens[begs[pi]];
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
            *cur_cens=pi;
            cur_cens++;
        }
    }
}

__global__
void computeNeighborIdxsRangeKernel(
        float * xyzs,               // [pn,3]
        int *begs,                  // [pn]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int* cur_idxs=&idxs[begs[pi]];
    int* cur_cens=&cens[begs[pi]];
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        float sq_dist=(tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz);
        if(sq_dist<squared_max_nn_size&&sq_dist>squared_min_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
            *cur_cens=pi;
            cur_cens++;
        }
    }
}


int searchNeighborhoodCountImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    countNeighborNumKernel<<<block_dim,thread_dim>>>(xyzs,lens,squared_nn_size,pn);
    gpuErrchk(cudaGetLastError())

    thrust::device_ptr<int> len_ptr(lens);
    thrust::device_ptr<int> beg_ptr(begs);
    thrust::exclusive_scan(len_ptr,len_ptr+pn,beg_ptr);
    gpuErrchk(cudaGetLastError())

    // todo: maybe error here?
    int count1,count2;
    gpuErrchk(cudaMemcpy(&count1,begs+pn-1,sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(&count2,lens+pn-1,sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaGetLastError())

    return count1+count2;
    // old version
    // return *(len_ptr+pn-1)+*(beg_ptr+pn-1);
}


int searchNeighborhoodCountRangeImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    countNeighborNumRangeKernel<<<block_dim,thread_dim>>>(xyzs,lens,squared_min_nn_size,squared_max_nn_size,pn);
    gpuErrchk(cudaGetLastError())

    thrust::device_ptr<int> len_ptr(lens);
    thrust::device_ptr<int> beg_ptr(begs);
    thrust::exclusive_scan(len_ptr,len_ptr+pn,beg_ptr);
    gpuErrchk(cudaGetLastError())

    // todo: maybe error here?
    int count1,count2;
    gpuErrchk(cudaMemcpy(&count1,begs+pn-1,sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaMemcpy(&count2,lens+pn-1,sizeof(int),cudaMemcpyDeviceToHost))
    gpuErrchk(cudaGetLastError())

    return count1+count2;
    // old version
    // return *(len_ptr+pn-1)+*(beg_ptr+pn-1);
}

void searchNeighborhoodImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsKernel<<<block_dim,thread_dim>>>(xyzs,begs,idxs,cens,squared_nn_size,pn);
    gpuErrchk(cudaGetLastError())
}

void searchNeighborhoodRangeImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        int *begs,                  // [pn]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsRangeKernel<<<block_dim,thread_dim>>>(xyzs,begs,idxs,cens,squared_min_nn_size,squared_max_nn_size,pn);
    gpuErrchk(cudaGetLastError())
}