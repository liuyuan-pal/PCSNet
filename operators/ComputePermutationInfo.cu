#include "CudaCommon.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <ctime>

__global__
void computeWholeIdxsKernel(
        int *voxel_idxs,    //[pn,3]
        long *voxel_whole_idxs,  // [pn]
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    int t=pi*3;
    voxel_whole_idxs[pi]=voxel_idxs[t];
    voxel_whole_idxs[pi]<<=16;
    voxel_whole_idxs[pi]+=voxel_idxs[t+1];
    voxel_whole_idxs[pi]<<=16;
    voxel_whole_idxs[pi]+=voxel_idxs[t+2];
}

__global__
void fillIndex(
        int *ptr,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    ptr[pi]=pi;
}

__global__
void computeCens(
        int *cens,
        int *lens,
        int *begs,
        int vn
)
{
    int vi = threadIdx.x + blockIdx.x*blockDim.x;
    if(vi>=vn) return;
    int* cur_cens=&cens[begs[vi]];
    int len=lens[vi];
    for(int i=0;i<len;i++)
    {
        *cur_cens=vi;
        cur_cens++;
    }

}


void computePermutationInfoImpl(
        int *voxel_idxs,                // [pn,3]
        int *origin2permutation_idxs,   // [pn]
        int **lens,                     // [vn]
        int **begs,                     // [vn]
        int **cens,                     // [vn]
        int *vn,
        int pn
)
{
//    time_t bg=clock();

    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    // get whole idx
    thrust::device_vector<long> whole_idxs(pn);
    computeWholeIdxsKernel<<<block_dim,thread_dim>>>(voxel_idxs,thrust::raw_pointer_cast(whole_idxs.data()),pn);
    gpuErrchk(cudaGetLastError())
    fillIndex<<<block_dim,thread_dim>>>(origin2permutation_idxs,pn);
    gpuErrchk(cudaGetLastError())

    // sort by keys get o2p idx
    thrust::device_ptr<int> o2p_ptr(origin2permutation_idxs);
    thrust::sort_by_key(thrust::device,whole_idxs.begin(),whole_idxs.end(),o2p_ptr);
    gpuErrchk(cudaGetLastError())

    thrust::device_vector<int> ones_array(pn);
    thrust::device_vector<int> result_keys(pn);
    thrust::device_vector<int> result_values(pn);
    thrust::fill(ones_array.begin(),ones_array.end(),1);
    gpuErrchk(cudaGetLastError())

    thrust::pair<thrust::detail::normal_iterator<thrust::device_ptr<int> >, thrust::detail::normal_iterator<thrust::device_ptr<int> > >
    new_end = thrust::reduce_by_key(thrust::device, whole_idxs.begin(), whole_idxs.end(),
                                   ones_array.begin(), result_keys.begin(), result_values.begin());
    gpuErrchk(cudaGetLastError())
    *vn=new_end.second-result_values.begin();

    gpuErrchk(cudaMalloc((void**)lens,sizeof(int)*(*vn)))
    thrust::device_ptr<int> lens_ptr(*lens);
    thrust::copy(result_values.data(),result_values.data()+(*vn),lens_ptr);
    gpuErrchk(cudaGetLastError())

    // scan begs
    gpuErrchk(cudaMalloc((void**)begs,sizeof(int)*(*vn)))
    thrust::device_ptr<int> begs_ptr(*begs);
    thrust::exclusive_scan(lens_ptr,lens_ptr+(*vn),begs_ptr);
    gpuErrchk(cudaGetLastError())

    // cens
    gpuErrchk(cudaMalloc((void**)cens,sizeof(int)*pn))
    int vblock_num = (*vn) / 1024;
    if ((*vn) % 1024 > 0) vblock_num++;
    dim3 vblock_dim(vblock_num);
    dim3 vthread_dim(1024);
    computeCens<<<vblock_dim,vthread_dim>>>(*cens,*lens,*begs,*vn);
    gpuErrchk(cudaGetLastError())
}


template<typename T>
void copyDataAndFree(
        T *in,
        T *out,
        int num
)
{
    gpuErrchk(cudaMemcpy(out,in,sizeof(T)*num,cudaMemcpyDeviceToDevice))
    gpuErrchk(cudaFree(in))
}


template void copyDataAndFree<int>(int*,int*,int);