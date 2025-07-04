#include "catch.hpp"
#include "smaug/core/backend.h"
#include "smaug/core/tensor.h"
#include "smaug/core/smaug_test.h"
#include "smaug/operators/smv/smv_test_common.h"
#include "smaug/operators/smv/smv_depthwise_convolution_op.h"
#include "smaug/operators/smv/smv_depthwise_convolution_tiling.h"
#include "smaug/operators/reorder_op.h"

 
using namespace smaug;

namespace smaug {


Workspace* work_space = new Workspace();

Tensor* getReferenceOutput(SmvDepthwiseConvolutionOp* convOp) {
        auto input = convOp->getInput(0);
        auto kernels = convOp->getInput(1);
        std::cout << " before creating reorder ops \n";
        auto input_reorder = new ReorderOp<SmvBackend>("input reorder",  work_space);
        auto kernel_reorder = new ReorderOp<SmvBackend>("kernel reorder",  work_space);
        auto output_reorder = new ReorderOp<SmvBackend>("output reorder",  work_space);
        std::cout << " after creating reorder ops\n";
        DataLayout dl = NCHW;
        input_reorder->setTargetLayout(dl);
        kernel_reorder->setTargetLayout(dl);
        dl = NHWC;
        output_reorder->setTargetLayout(dl);
        std::cout << " before setting reorder inputs \n";
        input_reorder->setInput(input,0);
        kernel_reorder->setInput(kernels,0);
        std::cout << " before creating reorder tensors \n";
        input_reorder -> createAllTensors();
        kernel_reorder->createAllTensors();
        std::cout << " after creating reorder tensors \n";
        input_reorder->getOutput(0)->allocateStorage<float16>();
        kernel_reorder->getOutput(0)->allocateStorage<float16>();
        
        std::cout<< " Reordering input and kernel \n";
        input_reorder->run();
        kernel_reorder->run();
        auto input_rd = input_reorder->getOutput(0);
        auto kernel_rd = kernel_reorder->getOutput(0);
        auto input32 = convertFp16ToFp32Tensor(input_rd, work_space);
        auto kernels32 = convertFp16ToFp32Tensor(kernel_rd, work_space);
        std::cout<<"Reorder completed \n";
        // A reference convolution operator is used to get the 'correct' output.
        auto refConvOp =
                new DepthwiseConvolutionOp<ReferenceBackend>("ref_dw_conv", work_space);
        refConvOp->setActivation(convOp->getActivation());
        refConvOp->setPadding(convOp->getPadding());
        refConvOp->setWeightDims(convOp->getWeightRows(),
                                 convOp->getWeightCols(),
                                 convOp->getNumOfmaps());
        refConvOp->setStride(convOp->getRowStride(), convOp->getColStride());
        refConvOp->setInput(input32, 0);
        refConvOp->setInput(kernels32, 1);
        refConvOp->createAllTensors();
        refConvOp->getOutput(0)->allocateStorage<float>();
        refConvOp->run();
        std::cout << " Completed dw_conv \n";
        auto output = convertFp32ToFp16Tensor(refConvOp->getOutput(0), work_space);
        output_reorder->setInput(output,0);
        //output_reorder->setInput(refConvOp->getOutput(0),0);

        output_reorder->createAllTensors();
        output_reorder->getOutput(0)->allocateStorage<float16>();

        output_reorder->run();
        std::cout<< "Completed output reordering \n";
        auto out_rd = output_reorder->getOutput(0);
        return out_rd;
    }

TEST_CASE_METHOD(SmaugTest, "SmvDepthwiseConvolutionOp", " [ops]") {
    PaddingType padding = SamePadding;
    TensorShape inputShape({1,56,56,144}, NHWC, SmvBackend::Alignment);
    Tensor* inputs = new Tensor("input", inputShape);
    TensorShape weightShape ({1,3,3,144}, NHWC,SmvBackend::Alignment);
    Tensor* weights = new Tensor("weights", weightShape);
    
    auto dw_conv_op = new SmvDepthwiseConvolutionOp("dw_conv",work_space);
    dw_conv_op->setStride (1,1);
    dw_conv_op -> setPadding(padding);
    inputs->allocateStorage<float16>();
    work_space->addTensor(inputs);
    weights-> allocateStorage<float16>();
    work_space->addTensor(weights);
    dw_conv_op->setInput(inputs, 0);
    dw_conv_op->setInput(weights, 1);
    dw_conv_op->setWeightDims(3,3,1);
    createAndFillTensorsWithData<float16>(dw_conv_op, fillTensorWithRandomData);
    dw_conv_op -> tile();
    std::cout << "Running the operator \n";
    dw_conv_op->run();
    std::cout << "Completed the operator \n";
    auto output = dw_conv_op->getOutput(0);
    std::cout << " got the output \n";
    //ref
    auto refOutputs = getReferenceOutput(dw_conv_op);
    std::cout << " got reference output \n";
    verifyOutputs<float16>(output, refOutputs);
    std::cout << " Verified the operator \n";
}

} //namespace smaug