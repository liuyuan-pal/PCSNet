#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;


template<typename FLT_TYPE,typename INT_TYPE>
void maxGather(
        FLT_TYPE *feats,             // [s,f]
        INT_TYPE *nlens,             // [m]
        INT_TYPE *nbegs,             // [m]
        INT_TYPE m,
        INT_TYPE f,
        FLT_TYPE *feats_gather,       // [m,f]
        INT_TYPE *idxs_gather         // [m,f]
);

REGISTER_OP("MaxGather")
    .Input("feats: float32") //[s,f]
    .Input("nlens: int32")   //[m]
    .Input("nbegs: int32")   //[m]
    .Input("ncens: int32")   //[s]
    .Output("feats_gather: float32")//[m,f]
    .Output("idxs_gather: int32")//[m,f]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle feats_shape;
        ::tensorflow::shape_inference::ShapeHandle nlens_shape;
        ::tensorflow::shape_inference::ShapeHandle nbegs_shape;
        ::tensorflow::shape_inference::ShapeHandle ncens_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nlens_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&nbegs_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&ncens_shape));

        std::initializer_list<shape_inference::DimensionOrConstant> dims=
                {c->Dim(nlens_shape,0),c->Dim(feats_shape,1)};
        c->set_output(0,c->MakeShape(dims));
        c->set_output(1,c->MakeShape(dims));
        return Status::OK();
    });


class MaxGatherOp: public OpKernel
{
public:
    explicit MaxGatherOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        const Tensor& feats=context->input(0);//[s,f]
        const Tensor& nlens=context->input(1);//[m]
        const Tensor& nbegs=context->input(2);//[m]
        const Tensor& ncens=context->input(3);//[s]

        int s=feats.dim_size(0),f=feats.dim_size(1),m=nlens.dim_size(0);

        OP_REQUIRES(context,ncens.dim_size(0)==s,errors::InvalidArgument("ncens dim 0"));
        OP_REQUIRES(context,nbegs.dim_size(0)==m,errors::InvalidArgument("nbegs dim 0"));

        std::initializer_list<int64> dims={m,f};
        Tensor *feats_gather=NULL;
        Tensor *idxs_gather=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&feats_gather));
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims),&idxs_gather));

        auto feats_p=const_cast<float*>(feats.shaped<float,2>({s,f}).data());
        auto nbegs_p=const_cast<int*>(nbegs.shaped<int,1>({m}).data());
        auto nlens_p=const_cast<int*>(nlens.shaped<int,1>({m}).data());
        auto feats_gather_p=feats_gather->shaped<float,2>({m,f}).data();
        auto idxs_gather_p=idxs_gather->shaped<int,2>({m,f}).data();

        maxGather(feats_p,nlens_p,nbegs_p,m,f,feats_gather_p,idxs_gather_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("MaxGather").Device(DEVICE_GPU), MaxGatherOp);