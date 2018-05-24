#include "CudaCommon.h"

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void sumGatherKernel(
        FLT_TYPE *feats,                // [s,f]
        INT_TYPE *nlens,                // [s]
        INT_TYPE *nbegs,                // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_gather          // [m,f]
)
{
    int mi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(mi>=m||fi>=f) return;
    INT_TYPE nn_bg=nbegs[mi];
    INT_TYPE nn_sz=nlens[mi];
    if(nn_sz==0) return;

    FLT_TYPE *feats_p=&feats[nn_bg*f+fi];
    FLT_TYPE *feats_gather_p=&feats_gather[mi*f+fi];
    for(int ni=0;ni<nn_sz;ni++)
    {
        (*feats_gather_p)+=(*feats_p);
        feats_p+=f;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
void sumGather(
        FLT_TYPE *feats,             // [s,f]
        INT_TYPE *nlens,             // [m]
        INT_TYPE *nbegs,             // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_gather       // [m,f]
)
{
    int tdim0,tdim1,tdim2,bdim0,bdim1,bdim2;
    getGPULayout(m,f,1,bdim0,bdim1,bdim2,tdim0,tdim1,tdim2);
    gpuErrchk(cudaMemset(feats_gather,0,m*f*sizeof(FLT_TYPE)))

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);
    sumGatherKernel<FLT_TYPE,INT_TYPE><<<block_dim,thread_dim>>>(feats,nlens,nbegs,m,f,feats_gather);
    gpuErrchk(cudaGetLastError())
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void maxGatherKernel(
        FLT_TYPE *feats,                // [s,f]
        INT_TYPE *nlens,                // [m]
        INT_TYPE *nbegs,                // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_gather,         // [m,f]
        INT_TYPE *idxs_gather           // [m,f] used in backward
)
{
    int mi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(mi>=m||fi>=f) return;
    INT_TYPE nn_bg=nbegs[mi];
    INT_TYPE nn_sz=nlens[mi];
    if(nn_sz==0) return;

    FLT_TYPE *feats_gather_p=&feats_gather[mi*f+fi];
    INT_TYPE *idxs_gather_p=&idxs_gather[mi*f+fi];
    FLT_TYPE *feats_p=&feats[nn_bg*f+fi];

    (*feats_gather_p)=*feats_p;
    (*idxs_gather_p)=0;
    feats_p+=f;
    for(int ni=1;ni<nn_sz;ni++)
    {
        FLT_TYPE cur_val=*feats_p;
        if((*feats_gather_p)<cur_val)
        {
            (*feats_gather_p)=cur_val;
            (*idxs_gather_p)=ni;
        }
        feats_p+=f;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
void maxGather(
        FLT_TYPE *feats,                // [s,f]
        INT_TYPE *nlens,                // [m]
        INT_TYPE *nbegs,                // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_gather,         // [m,f]
        INT_TYPE *idxs_gather           // [m,f] used in backward
)
{
    int tdim0,tdim1,tdim2,bdim0,bdim1,bdim2;
    getGPULayout(m,f,1,bdim0,bdim1,bdim2,tdim0,tdim1,tdim2);
    gpuErrchk(cudaMemset(feats_gather,0,m*f*sizeof(FLT_TYPE)))
    gpuErrchk(cudaMemset(idxs_gather,0,m*f*sizeof(INT_TYPE)))

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);
    maxGatherKernel<FLT_TYPE,INT_TYPE><<<block_dim,thread_dim>>>(feats,nlens,nbegs,m,f,feats_gather,idxs_gather);
    gpuErrchk(cudaGetLastError())
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void maxScatterKernel(
        FLT_TYPE *feats,                // [m,f]
        INT_TYPE *idxs,                 // [m,f]
        INT_TYPE *nlens,                // [m]
        INT_TYPE *nbegs,                // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_scatter         // [s,f]
)
{
    int mi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(mi>=m||fi>=f||nlens[mi]==0) return;

    INT_TYPE ni=nbegs[mi]+idxs[mi*f+fi];
    feats_scatter[ni*f+fi]=feats[mi*f+fi];
}


template<typename FLT_TYPE,typename INT_TYPE>
void maxScatter(
        FLT_TYPE *feats,                // [m,f]
        INT_TYPE *idxs,                 // [m,f]
        INT_TYPE *nlens,                // [m]
        INT_TYPE *nbegs,                // [m]
        INT_TYPE m,
        INT_TYPE s,
        INT_TYPE f,
        FLT_TYPE *feats_scatter         // [s,f]
)
{
    int tdim0,tdim1,tdim2,bdim0,bdim1,bdim2;
    getGPULayout(m,f,1,bdim0,bdim1,bdim2,tdim0,tdim1,tdim2);
    gpuErrchk(cudaMemset(feats_scatter,0,s*f*sizeof(FLT_TYPE)))

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);
    maxScatterKernel<FLT_TYPE,INT_TYPE><<<block_dim,thread_dim>>>(feats,idxs,nlens,nbegs,m,f,feats_scatter);
    gpuErrchk(cudaGetLastError())
}

template void sumGather<float,int>(float*,int*,int*,int,int,float*);
template void maxGather<float,int>(float*,int*,int*,int,int,float*,int*);
template void maxScatter<float,int>(float*,int*,int*,int*,int,int,int,float*);