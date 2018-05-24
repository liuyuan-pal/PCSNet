//
// Created by pal on 18-3-29.
//


#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>
#include <vector>

using namespace tensorflow;

void computeIdxsMap(
        int * voxel_idxs,
        int pn,
        std::vector<int>& origin2permutation_idxs,
        std::vector<int>& voxel_idxs_lens
);

REGISTER_OP("ComputePermutationInfo")
        .Input("voxel_idxs: int32")                     // [pn1,3]
        .Output("origin2permutation_idxs: int32")       // [pn1]
        .Output("voxel_idxs_lens: int32")               // [pn2]
        .Output("voxel_idxs_begs: int32")               // [pn2]
        .Output("voxel_idxs_cens: int32")               // [pn1]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle voxel_idxs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&voxel_idxs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims1=
                    {c->Dim(voxel_idxs_shape,0)};
            std::initializer_list<shape_inference::DimensionOrConstant> dims2=
                    {c->UnknownDim()};
            c->set_output(0,c->MakeShape(dims1));
            c->set_output(1,c->MakeShape(dims2));
            c->set_output(2,c->MakeShape(dims2));
            c->set_output(3,c->MakeShape(dims1));
            return Status::OK();
        });


class ComputePermutationInfoCPUOp: public OpKernel
{
public:
    explicit ComputePermutationInfoCPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& voxel_idxs=context->input(0);      // [pn1,fd]

        unsigned int pn=voxel_idxs.dim_size(0);
        OP_REQUIRES(context,voxel_idxs.dim_size(1)==3,errors::InvalidArgument("voxel_idxs dim 1"));
        auto voxel_idxs_data= const_cast<int*>(voxel_idxs.shaped<int,2>({pn,3}).data());


        std::vector<int> origin2permutation_idxs_vec,voxel_idxs_lens_vec;
        computeIdxsMap(voxel_idxs_data,pn,origin2permutation_idxs_vec,voxel_idxs_lens_vec);
        int vn=voxel_idxs_lens_vec.size();
        std::vector<int> voxel_idxs_begs_vec(vn);
        std::vector<int> voxel_idxs_cens_vec(pn);
        voxel_idxs_begs_vec[0]=0;
        for(int i=1;i<vn;i++)
            voxel_idxs_begs_vec[i]=voxel_idxs_lens_vec[i-1]+voxel_idxs_begs_vec[i-1];

        for(int i=0;i<vn;i++)
            for(int j=0;j<voxel_idxs_lens_vec[i];j++)
                voxel_idxs_cens_vec[voxel_idxs_begs_vec[i]+j]=i;

        std::initializer_list<int64> dims1={pn};
        std::initializer_list<int64> dims2={vn};
        Tensor *origin2permutation_idxs,
                *voxel_idxs_lens,
                *voxel_idxs_begs,
                *voxel_idxs_cens;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims1),&origin2permutation_idxs));
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims2),&voxel_idxs_lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims2),&voxel_idxs_begs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dims1),&voxel_idxs_cens));
        auto origin2permutation_idxs_data=const_cast<int*>(origin2permutation_idxs->shaped<int,1>({pn}).data());
        auto voxel_idxs_lens_data=const_cast<int*>(voxel_idxs_lens->shaped<int,1>({vn}).data());
        auto voxel_idxs_begs_data=const_cast<int*>(voxel_idxs_begs->shaped<int,1>({vn}).data());
        auto voxel_idxs_cens_data=const_cast<int*>(voxel_idxs_cens->shaped<int,1>({pn}).data());
        std::copy(origin2permutation_idxs_vec.begin(),origin2permutation_idxs_vec.end(),origin2permutation_idxs_data);
        std::copy(voxel_idxs_lens_vec.begin(),voxel_idxs_lens_vec.end(),voxel_idxs_lens_data);
        std::copy(voxel_idxs_begs_vec.begin(),voxel_idxs_begs_vec.end(),voxel_idxs_begs_data);
        std::copy(voxel_idxs_cens_vec.begin(),voxel_idxs_cens_vec.end(),voxel_idxs_cens_data);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputePermutationInfo").Device(DEVICE_CPU), ComputePermutationInfoCPUOp);


void computePermutationInfoImpl(
        int *voxel_idxs,                // [pn,3]
        int *origin2permutation_idxs,   // [pn]
        int **lens,                     // [vn]
        int **begs,                     // [vn]
        int **cens,                     // [vn]
        int *vn,
        int pn
);

template<typename T>
void copyDataAndFree(
        T *in,
        T *out,
        int num
);

class ComputePermutationInfoGPUOp: public OpKernel
{
public:
    explicit ComputePermutationInfoGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& voxel_idxs=context->input(0);      // [pn1,3]

        unsigned int pn=voxel_idxs.dim_size(0);
        OP_REQUIRES(context,voxel_idxs.dim_size(1)==3,errors::InvalidArgument("voxel_idxs dim 1"));
        auto voxel_idxs_data= const_cast<int*>(voxel_idxs.shaped<int,2>({pn,3}).data());

        std::initializer_list<int64> dims1={pn};
        Tensor *origin2permutation_idxs,
                *voxel_idxs_lens,
                *voxel_idxs_begs,
                *voxel_idxs_cens;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims1),&origin2permutation_idxs));
        auto origin2permutation_idxs_data=const_cast<int*>(origin2permutation_idxs->shaped<int,1>({pn}).data());
        int *lens,*begs,*cens,vn;
        computePermutationInfoImpl(voxel_idxs_data,origin2permutation_idxs_data,&lens,&begs,&cens,&vn,pn);

        std::initializer_list<int64> dims2={vn};
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims2),&voxel_idxs_lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims2),&voxel_idxs_begs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dims1),&voxel_idxs_cens));
        auto voxel_idxs_lens_data=const_cast<int*>(voxel_idxs_lens->shaped<int,1>({vn}).data());
        auto voxel_idxs_begs_data=const_cast<int*>(voxel_idxs_begs->shaped<int,1>({vn}).data());
        auto voxel_idxs_cens_data=const_cast<int*>(voxel_idxs_cens->shaped<int,1>({pn}).data());

        copyDataAndFree(lens, voxel_idxs_lens_data, vn);
        copyDataAndFree(begs, voxel_idxs_begs_data, vn);
        copyDataAndFree(cens, voxel_idxs_cens_data, pn);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputePermutationInfo").Device(DEVICE_GPU), ComputePermutationInfoGPUOp);