//
// Created by pal on 18-3-29.
//


#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <initializer_list>
#include <vector>

//#include <log4cpp/Category.hh>
//#include <log4cpp/FileAppender.hh>
//#include <log4cpp/BasicLayout.hh>
//#include <log4cpp/Priority.hh>

using namespace tensorflow;


REGISTER_OP("ComputeRepermutationInfo")
        .Input("o2p_idxs2: int32")                     // [pn2]
        .Input("lens: int32")                          // [pn2]
        .Input("begs: int32")                          // [pn2]
        .Input("cens: int32")                          // [pn1]
        .Output("reper_o2p_idxs1: int32")              // [pn1]
        .Output("reper_lens: int32")                   // [pn2]
        .Output("reper_begs: int32")                   // [pn2]
        .Output("reper_cens: int32")                   // [pn1]
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle o2p_idxs2_shape;
            ::tensorflow::shape_inference::ShapeHandle lens_shape;
            ::tensorflow::shape_inference::ShapeHandle begs_shape;
            ::tensorflow::shape_inference::ShapeHandle cens_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),1,&o2p_idxs2_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&lens_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),1,&begs_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3),1,&cens_shape));

            c->set_output(0,cens_shape);
            c->set_output(1,o2p_idxs2_shape);
            c->set_output(2,o2p_idxs2_shape);
            c->set_output(3,cens_shape);
            return Status::OK();
        });

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
);

class ComputeRepermutationInfoGPUOp: public OpKernel
{
public:
    explicit ComputeRepermutationInfoGPUOp(OpKernelConstruction* context) : OpKernel(context)
    {
    }
    void Compute(OpKernelContext* context) override
    {
//        log4cpp::Appender *fileAppender = new log4cpp::FileAppender("fa", "./test.log");
//        fileAppender->setLayout(new log4cpp::BasicLayout());
//
//        log4cpp::Category& root =log4cpp::Category::getRoot();
//        root.addAppender(fileAppender);
//        root.setPriority(log4cpp::Priority::DEBUG);
//        root.warn("ComputeRepermutationInfoGPUOp Begin");
//        std::cout<<"ComputePermutationInfoWithClassGPUOp begin\n";
        // fetch input tensor
        const Tensor& o2p_idxs2=context->input(0);      // [pn2]
        const Tensor& lens=context->input(1);           // [pn2]
        const Tensor& begs=context->input(2);           // [pn2]
        const Tensor& cens=context->input(3);           // [pn1]

        unsigned int pn1=cens.dim_size(0),
                     pn2=o2p_idxs2.dim_size(0);
        OP_REQUIRES(context,lens.dim_size(0)==pn2,errors::InvalidArgument("lens dim 0"));
        OP_REQUIRES(context,begs.dim_size(0)==pn2,errors::InvalidArgument("begs dim 0"));

        auto o2p_idxs2_data= const_cast<int*>(o2p_idxs2.shaped<int,1>({pn2}).data());
        auto lens_data= const_cast<int*>(lens.shaped<int,1>({pn2}).data());
        auto begs_data= const_cast<int*>(begs.shaped<int,1>({pn2}).data());

        std::initializer_list<int64> dims1={pn1};
        std::initializer_list<int64> dims2={pn2};
        Tensor *reper_o2p_idxs1,
                *reper_lens,
                *reper_begs,
                *reper_cens;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape(dims1),&reper_o2p_idxs1));
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape(dims2),&reper_lens));
        OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape(dims2),&reper_begs));
        OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape(dims1),&reper_cens));

        auto reper_o2p_idxs1_data=const_cast<int*>(reper_o2p_idxs1->shaped<int,1>({pn1}).data());
        auto reper_lens_data=const_cast<int*>(reper_lens->shaped<int,1>({pn2}).data());
        auto reper_begs_data=const_cast<int*>(reper_begs->shaped<int,1>({pn2}).data());
        auto reper_cens_data=const_cast<int*>(reper_cens->shaped<int,1>({pn1}).data());
        computeRepermutationInfoImpl(o2p_idxs2_data,lens_data,begs_data,reper_lens_data,reper_begs_data,
                                     reper_cens_data,reper_o2p_idxs1_data,pn1,pn2);

//        root.warn("ComputeRepermutationInfoGPUOp End");
//        log4cpp::Category::shutdown();
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputeRepermutationInfo").Device(DEVICE_GPU), ComputeRepermutationInfoGPUOp);