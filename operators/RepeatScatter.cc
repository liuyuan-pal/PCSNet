#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;


template<typename FLT_TYPE,typename INT_TYPE>
void repeatScatter(
        FLT_TYPE *feats,                // [m,f]
        INT_TYPE *ncens,                // [s]
        INT_TYPE s,
        INT_TYPE f,
        FLT_TYPE *feats_scatter         // [s,f]
);

REGISTER_OP("RepeatScatter")
    .Input("feats: float32") //[m,f]
    .Input("nlens: int32")   //[s]
    .Input("nbegs: int32")   //[s]
    .Input("ncens: int32")   //[s]
    .Output("feats_scatter: float32")//[s,f]
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle feats_shape;
        ::tensorflow::shape_inference::ShapeHandle nlens_shape;
        ::tensorflow::shape_inference::ShapeHandle nbegs_shape;
        ::tensorflow::shape_inference::ShapeHandle ncens_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feats_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nlens_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&nbegs_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&ncens_shape));

        std::initializer_list<shape_inference::DimensionOrConstant> dims=
                {c->Dim(ncens_shape,0),c->Dim(feats_shape,1)};
        c->set_output(0,c->MakeShape(dims));
        return Status::OK();
    });


class RepeatScatterOp: public OpKernel
{
public:
    explicit RepeatScatterOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }
    void Compute(OpKernelContext* context) override
    {
        const Tensor& feats=context->input(0);//[m,f]
        const Tensor& nlens=context->input(1);//[m]
        const Tensor& nbegs=context->input(2);//[m]
        const Tensor& ncens=context->input(3);//[s]

        int m=feats.dim_size(0),f=feats.dim_size(1),s=ncens.dim_size(0);

        OP_REQUIRES(context,nlens.dim_size(0)==m,errors::InvalidArgument("nlens dim 0"));
        OP_REQUIRES(context,nbegs.dim_size(0)==m,errors::InvalidArgument("nbegs dim 0"));

        std::initializer_list<int64> dims={s,f};
        Tensor *feats_scatter=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims),&feats_scatter));

        auto feats_p=const_cast<float*>(feats.shaped<float,2>({m,f}).data());
        auto ncens_p=const_cast<int*>(ncens.shaped<int,1>({s}).data());
        auto feats_scatter_p=feats_scatter->shaped<float,2>({s,f}).data();

        repeatScatter(feats_p,ncens_p,s,f,feats_scatter_p);
    }
};

REGISTER_KERNEL_BUILDER(Name("RepeatScatter").Device(DEVICE_GPU), RepeatScatterOp);