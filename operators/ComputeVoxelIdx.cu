#include "CudaCommon.h"

__global__
void computeVoxelIdxKernel(
        float *pts,
        float *min_xyz,
        unsigned int *voxel_idxs,
        int pt_num,
        int pt_stride,
        float voxel_len
)
{
    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>=pt_num)
        return;

    float x=pts[pt_index*pt_stride];
    float y=pts[pt_index*pt_stride+1];
    float z=pts[pt_index*pt_stride+2];

    float eps=1e-3;

    float min_x=min_xyz[0]+eps;
    float min_y=min_xyz[1]+eps;
    float min_z=min_xyz[2]+eps;

    voxel_idxs[pt_index*3] = floor((x-min_x)/voxel_len);
    voxel_idxs[pt_index*3+1] = floor((y-min_y)/voxel_len);
    voxel_idxs[pt_index*3+2] = floor((z-min_z)/voxel_len);

}

void computeVoxelIdxImpl(
        float* pts,
        float* min_xyz,
        unsigned int* voxel_idxs,
        int pt_num,
        int pt_stride,
        float voxel_len
)
{
    int block_num=pt_num/1024;
    if(pt_num%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    computeVoxelIdxKernel<<<block_dim,thread_dim>>>(
           pts,min_xyz,voxel_idxs,pt_num,pt_stride,voxel_len
    );
    gpuErrchk(cudaGetLastError())
}