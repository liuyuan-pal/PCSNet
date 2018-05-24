//
// Created by pal on 18-3-29
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>

using namespace tensorflow;

int searchNeighborhoodCountImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
);

void searchNeighborhoodImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
);


REGISTER_OP("SearchNeighborhood")
        .Input("xyzs: float32")                 // [pn]
        .Attr("squared_nn_size: float")         // [pn]
        .Output("idxs: int32")     // [en]
        .Output("lens: int32")     // [pn]
        .Output("begs: int32")     // [pn]
        .Output("cens: int32")     // [en]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle xyzs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&xyzs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(xyzs_shape,0)};
            std::initializer_list<shape_inference::DimensionOrConstant> unknow_dims=
                    {c->UnknownDim()};
            c->set_output(0,c->MakeShape(unknow_dims));
            c->set_output(1,c->MakeShape(dims));
            c->set_output(2,c->MakeShape(dims));
            c->set_output(3,c->MakeShape(unknow_dims));
            return Status::OK();
        });

class SearchNeighborhoodGPUOp: public OpKernel
{
    float squared_nn_size;
public:
    explicit SearchNeighborhoodGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("squared_nn_size",&squared_nn_size);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& xyzs=context->input(0);       // [pn1,3]

        unsigned int pn=xyzs.dim_size(0);
        OP_REQUIRES(context,xyzs.dim_size(1)==3,errors::InvalidArgument("xyzs dim 1"));

        std::initializer_list<int64> dims={pn};
        Tensor *idxs,*lens,*begs,*cens;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims),&lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims),&begs));
        auto xyzs_data=const_cast<float*>(xyzs.shaped<float,2>({pn,3}).data());
        auto lens_data=const_cast<int*>(lens->shaped<int,1>({pn}).data());
        auto begs_data=const_cast<int*>(begs->shaped<int,1>({pn}).data());

        int en=searchNeighborhoodCountImpl(xyzs_data,lens_data,begs_data,squared_nn_size,pn);

        std::initializer_list<int64> dims2={en};
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims2),&idxs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dims2),&cens));
        auto idxs_data=const_cast<int*>(idxs->shaped<int,1>({en}).data());
        auto cens_data=const_cast<int*>(cens->shaped<int,1>({en}).data());
        searchNeighborhoodImpl(xyzs_data,idxs_data,cens_data,begs_data,squared_nn_size,pn);
    }
};

REGISTER_KERNEL_BUILDER(Name("SearchNeighborhood").Device(DEVICE_GPU), SearchNeighborhoodGPUOp);

int searchNeighborhoodCountRangeImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
);

void searchNeighborhoodRangeImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        int *begs,                  // [pn]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int pn
);


REGISTER_OP("SearchNeighborhoodRange")
        .Input("xyzs: float32")                 // [pn]
        .Attr("squared_min_nn_size: float")         // [pn]
        .Attr("squared_max_nn_size: float")         // [pn]
        .Output("idxs: int32")     // [en]
        .Output("lens: int32")     // [pn]
        .Output("begs: int32")     // [pn]
        .Output("cens: int32")     // [en]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle xyzs_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&xyzs_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(xyzs_shape,0)};
            std::initializer_list<shape_inference::DimensionOrConstant> unknow_dims=
                    {c->UnknownDim()};
            c->set_output(0,c->MakeShape(unknow_dims));
            c->set_output(1,c->MakeShape(dims));
            c->set_output(2,c->MakeShape(dims));
            c->set_output(3,c->MakeShape(unknow_dims));
            return Status::OK();
        });

class SearchNeighborhoodRangeGPUOp: public OpKernel
{
    float squared_min_nn_size;
    float squared_max_nn_size;
public:
    explicit SearchNeighborhoodRangeGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("squared_min_nn_size",&squared_min_nn_size);
        context->GetAttr("squared_max_nn_size",&squared_max_nn_size);
    }
    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& xyzs=context->input(0);       // [pn1,3]

        unsigned int pn=xyzs.dim_size(0);
        OP_REQUIRES(context,xyzs.dim_size(1)==3,errors::InvalidArgument("xyzs dim 1"));

        std::initializer_list<int64> dims={pn};
        Tensor *idxs,*lens,*begs,*cens;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims),&lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims),&begs));
        auto xyzs_data=const_cast<float*>(xyzs.shaped<float,2>({pn,3}).data());
        auto lens_data=const_cast<int*>(lens->shaped<int,1>({pn}).data());
        auto begs_data=const_cast<int*>(begs->shaped<int,1>({pn}).data());

        int en=searchNeighborhoodCountRangeImpl(xyzs_data,lens_data,begs_data,squared_min_nn_size,squared_max_nn_size,pn);

        std::initializer_list<int64> dims2={en};
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims2),&idxs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dims2),&cens));
        auto idxs_data=const_cast<int*>(idxs->shaped<int,1>({en}).data());
        auto cens_data=const_cast<int*>(cens->shaped<int,1>({en}).data());
        searchNeighborhoodRangeImpl(xyzs_data,idxs_data,cens_data,begs_data,squared_min_nn_size,squared_max_nn_size,pn);
    }
};

REGISTER_KERNEL_BUILDER(Name("SearchNeighborhoodRange").Device(DEVICE_GPU), SearchNeighborhoodRangeGPUOp);
