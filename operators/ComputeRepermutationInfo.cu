#include "CudaCommon.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>


__global__
void computeCens(
        int *cens,
        int *lens,
        int *begs,
        int vn
);

__global__
void computeRepermuatedIdxsKernel(
        int* o2p_idxs2,    // [pn2]
        int* begs,         // [pn2]
        int* reper_lens,   // [pn2]
        int* reper_begs,   //
        int* reper_o2p_idxs1,
        int pn2
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn2) return;
    int clen=reper_lens[pi];
    int obeg=begs[o2p_idxs2[pi]],rbeg=reper_begs[pi];
    int* rloc=&reper_o2p_idxs1[rbeg];
    for(int i=0;i<clen;i++)
        rloc[i]=obeg+i;
}

__global__
void computeRepermuatedLens(
        int* o2p_idxs2,    // [pn2]
        int* lens,         // [pn2]
        int* reper_lens,   // [pn2]
        int pn2
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn2) return;
    reper_lens[pi]=lens[o2p_idxs2[pi]];
}

void computeRepermutationInfoImpl(
        int* o2p_idxs2,    // [pn2]
        int* lens,         // [pn2]
        int* begs,         // [pn2]

        int* reper_lens,         // [pn2]
        int* reper_begs,         // [pn2]
        int* reper_cens,         // [pn2]
        int* reper_o2p_idxs1,    // [pn1]
        int pn1,
        int pn2
)
{

    int block_num = pn2 / 1024;
    if (pn2 % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeRepermuatedLens<<<block_dim,thread_dim>>>(o2p_idxs2,lens,reper_lens,pn2);

    thrust::device_ptr<int> reper_begs_ptr(reper_begs);
    thrust::device_ptr<int> reper_lens_ptr(reper_lens);
    thrust::exclusive_scan(reper_lens_ptr,reper_lens_ptr+pn2,reper_begs_ptr);
    computeRepermuatedIdxsKernel<<<block_dim,thread_dim>>>(o2p_idxs2,begs,reper_lens,reper_begs,reper_o2p_idxs1,pn2);
    gpuErrchk(cudaGetLastError())

    int block_num2 = pn1 / 1024;
    if (pn1 % 1024 > 0) block_num2++;
    dim3 block_dim2(block_num2);
    dim3 thread_dim2(1024);

    computeCens<<<block_dim,thread_dim>>>(reper_cens,reper_lens,reper_begs,pn2);
    gpuErrchk(cudaGetLastError())
}


