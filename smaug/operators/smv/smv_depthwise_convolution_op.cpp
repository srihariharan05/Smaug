#include "smaug/core/backend.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_depthwise_convolution_op.h"
#include "smaug/operators/smv/smv_depthwise_convolution_tiling.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/operators/smv/smv_accel_pool.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {
namespace smv {
namespace dwconv {

const int VectorSize = 8;
#if (ACC_1_TOPS == 1) 
    const int kNumPEs = 32; //8;
    const int kNumMaccsPerPE = 4;
#elif (ACC_32_TOPS ==1 )
    const int kNumPEs = 256;
    const int kNumMaccsPerPE = 16;
#elif (ACC_256_GOPS == 1)
    const int kNumPEs = 8;
    const int kNumMaccsPerPE = 4;
#endif

}  // namespace conv
}  // namespace smv

void SmvDepthwiseConvolutionOp::runNHWC(TiledTensor& inputs,
                               TiledTensor& weights,
                               TiledTensor& outputs) {
    int inputIfmapTiles = inputs.getShape()[0];
    int inputRowTiles = inputs.getShape()[1];
    int inputChanTiles = inputs.getShape()[3];
    int weightOfmapTiles = weights.getShape()[0];
    int weightChanTiles = weights.getShape()[3];
    int outputRowTiles = outputs.getShape()[1];
    int outputChanTiles = outputs.getShape()[3];
    auto inputIdx = inputs.startIndex();
    auto weightIdx = weights.startIndex();
    auto outputIdx = outputs.startIndex();
    std::vector<int> inputPadding = getInputPadding();
    int topPad = inputPadding[0];
    int bottomPad = inputPadding[1];
    int leftPad = inputPadding[2];
    int rightPad = inputPadding[3];
    unsigned accelId = smv::kDepthwiseConvolutionHw;
    SmvAcceleratorPool accelPool(numAcceleratorsAvailable);
    std::vector<int> lastReadInputTileIdx(numAcceleratorsAvailable, -1);
    std::vector<int> lastReadWeightTileIdx(numAcceleratorsAvailable, -1);
    for (int i = 0; i < numAcceleratorsAvailable; i++) {
        setArrayMemTypeIfSimulating(
                accelId + i, "host_inputs", getInputsMemType());
        setArrayMemTypeIfSimulating(
                accelId + i, "host_weights", getWeightsMemType());
        setArrayMemTypeIfSimulating(
                accelId + i, "host_results", getOutputsMemType());
    }
    int currAccelIdx = 0;
    int outputTileIdx = 0;
    int output_channel_offset =0;
    for (int N = 0; N < inputIfmapTiles; N++) {
        for (int H = 0; H < outputRowTiles; H++) {
            int currentTileTopPad = topPad;
            int currentTileBottomPad = bottomPad;
            if (inputRowTiles > 1) {
                if (H == 0) {
                    currentTileBottomPad = 0;
                } else if (H == inputRowTiles - 1) {
                    currentTileTopPad = 0;
                } else {
                    currentTileTopPad = 0;
                    currentTileBottomPad = 0;
                }
            }
            // This is used to specify the padding sizes on the boundaries of
            // the 2D feature maps in an input tile.
            int inputHaloPad[4] = { currentTileTopPad, currentTileBottomPad,
                                    leftPad, rightPad };
            
            bool needOutputIteration = weightChanTiles < outputChanTiles;
            // This is the number of invocations we need to finish the weight
            // tile. In common scenarios, only one invocation is needed. If we
            // need to iterate the output channels, outputChanTiles invocatons
            // are needed to finish the weight tile.
            int numOutputInvocations = 1; 
                   // needOutputIteration ? (outputChanTiles/weightChanTiles) : 1;
           
            int wC =0, iC =0; 
            int ifmapOffset = 0;
            output_channel_offset =0;
            while ( wC < weightChanTiles && iC < inputChanTiles) {
                int kernStart = 0;
                int inputTileIdx = inputIdx(N, H, 0, iC);
                int weightTileIdx = weightIdx(0, 0, 0, wC);
                //std::cout << " getting input tile : iteration : " << N << " " << H << " " << wC <<" " << iC << " \n";
                
                inputs[inputTileIdx]->allocateStorage(Float16);
                weights[weightTileIdx]->allocateStorage(Float16);

                Tensor* inputTile =
                        inputs.getTileWithData(inputTileIdx);
                Tensor* weightsTile =
                        weights.getTileWithData(weightTileIdx);
                //std::cout << " Tiles obained : iteration : " << N << " " << H << " " << wC <<" " << iC << " \n";

                const TensorShape& inputShape = inputTile->getShape();
                const TensorShape& weightsShape =
                        weightsTile->getShape();
                          
                mapArrayToAccel(
                        accelId + currAccelIdx, "host_inputs",
                        inputTile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
                mapArrayToAccel(
                        accelId + currAccelIdx, "host_weights",
                        weightsTile->data<float16>(),
                        weightsShape.storageSize() * sizeof(float16));
                /*flush_memory((void*)inputTile->data<float16>(),inputShape.storageSize() * sizeof(float16));
                flush_memory((void*)weightsTile->data<float16>(),weightsShape.storageSize() * sizeof(float16));*/
                output_channel_offset = wC * numOutputInvocations;
                int out_tile_id = outputIdx(N,H,0, output_channel_offset);
                int out_first_tile_shape = outputs[out_tile_id]->getShape()[3];
                numOutputInvocations = FRAC_CEIL(weightsShape[3], out_first_tile_shape ); 
                if ( !needOutputIteration)
                    numOutputInvocations = 1;
                assert(numOutputInvocations == 1 ? (out_first_tile_shape == weightsShape[3]): 1);
                for (int oC = 0; oC < numOutputInvocations; oC++) {
                    //int iC = 0, wC = 0;
                    // This keeps track of the channel offset of the input.
                    outputTileIdx = outputIdx(N, H, 0, (output_channel_offset + oC));
                    assert ( ((weightsShape[3] - (out_first_tile_shape * oC))  > 0) && "Output tile iterating beyond available weight tiles"); 
                    //std::cout << " getting output tile : iteration : " << N << " " << H << " " << wC <<" " << iC <<  " "<< oC << " \n";
                    //tiledTensors[2].getTile(outputTileIdx)->tensor->allocateStorage(tiledTensors[2].getTile(outputTileIdx)->tensor->getDataType());
                    Tensor* outputTile = outputs[outputTileIdx];
                    //outputTile->allocateStorage(getOutput(Outputs)->getDataType());
                    outputTile->allocateStorage(Float16);

                    const TensorShape& outputShape = outputTile->getShape();
                    mapArrayToAccel(
                            accelId + currAccelIdx, "host_results",
                            outputTile->data<float16>(),
                            outputShape.storageSize() * sizeof(float16));
                    //flush_memory((void*)outputTile->data<float16>(),outputShape.storageSize() * sizeof(float16));

                    
                    dout(1) << "Input: " << inputTileIdx
                        << ", weights: " << weightTileIdx
                        << ", output: " << outputTileIdx << "\n";
                    int inputDims[4] = { inputShape[0], inputShape[1],
                                            inputShape[2], inputShape[3] };
                    int weightsDims[4] = { weightsShape[0], weightsShape[1],
                                            weightsShape[2],
                                            weightsShape[3] };
                    int outputDims[4] = { outputShape[0], outputShape[1],
                                            outputShape[2], outputShape[3] };
                    // The 'ifmap_start' argument of the kernel is for
                    // handling when inputChanTiles < weightChanTiles. It
                    // provides the starting channel of the input tile that
                    // will be effective for computation in the invocation.
                    int ifmapStart = (iC == wC) ? 0 : ifmapOffset;
                    // Since multiple weight channelwise tiles produce the
                    // same output channels, 'accumulate' is set to true to
                    // avoid resetting the result for non-first (wC > 0)
                    // weight channelwise tiles.
                    ifmapStart += kernStart; 
                    bool accumulate = 0; //wC > 0;ssss
                    // If this is a new input/weight tile, then we need to
                    // read it.
                    bool readInputs = false;
                    if (inputTileIdx !=
                        lastReadInputTileIdx[currAccelIdx]) {
                        readInputs = true;
                        lastReadInputTileIdx[currAccelIdx] = inputTileIdx;
                    }
                    bool readWeights = false;
                    if (weightTileIdx !=
                        lastReadWeightTileIdx[currAccelIdx]) {
                        readWeights = true;
                        lastReadWeightTileIdx[currAccelIdx] = weightTileIdx;
                    }
                    // If we reach the last invocation for the weight
                    // channelwise tiles, the results are finished and need
                    // to be sent back to the host.
                    bool sendResults = 1; //wC == weightChanTiles - 1;

                    std::unique_ptr<volatile int> finishFlag;
                        //std::cout << " running the dw_conv kernel : iteration: " << N << " "<< H << " "<< wC << " " << iC << " " << oC <<" \n";
                        finishFlag = invokeKernelNoBlock(
                            currAccelIdx, accelId + currAccelIdx,
                                smv_depthwise_conv3d_nhwc_vec_fxp,
                                inputTile->data<float16>(),
                                weightsTile->data<float16>(),
                                outputTile->data<float16>(), smv::spad0,
                                smv::spad1, smv::spad2, inputDims,
                                weightsDims, outputDims,
                                inputShape.getPadding(3),
                                weightsShape.getPadding(3),
                                outputShape.getPadding(3), inputHaloPad,
                                getRowStride(), getColStride(), ifmapStart,
                                kernStart, accumulate, readInputs,
                                readWeights, sendResults, actInfo.function,
                                actInfo.params, &sampling);
                    //}
                    //std::cout << " Finished running the kernel \n";
                    accelPool.addFinishFlag(
                            currAccelIdx, std::move(finishFlag));

                    if (needOutputIteration)
                        kernStart += outputShape[3];
                }
                ifmapOffset += weightsTile->getShape()[3];
                if (inputChanTiles == weightChanTiles) {
                    iC++;
                    wC++;
                } else if (inputChanTiles == 1) {
                    wC++;
                } else {
                    assert(false &&
                            "The input/weight tiles can have different "
                            "number of channels only when the inputs "
                            "don't need channelwise tiling.");
                }
        }
    }
                currAccelIdx =
                        accelPool.getNextAvailableAccelerator(currAccelIdx);
            }
    // Before we leave, make sure all the accelerators have finished.
    accelPool.joinAll();
}

/*std::unique_ptr<volatile int> SmvDepthwiseConvolutionOp::invokeSystolicArrayKernel(
        unsigned accelId,
        float16* inputs,
        float16* weights,
        float16* outputs,
        int inputsDims[4],
        int weightsDims[4],
        int outputsDims[4],
        int inputsPad,
        int weightsPad,
        int outputPad,
        int inputHaloPad[4],
        int stride,
        int ifmapStart,
        int kernStart,
        bool accumulate,
        bool readInputs,
        bool readWeights,
        bool sendResults,
        ActivationInfo* actInfo) {
    // Note that if we are in trace mode, we should skip this gem5 accelerator.
#ifndef TRACE_MODE
    assert(runningInSimulation && "The systolic array must be invoked in "
                                  "simuation.");
    systolic_array_params_t params;
    params.input_base_addr = inputs;
    params.weight_base_addr = weights;
    params.output_base_addr = outputs;
    memcpy(params.input_dims, inputsDims, sizeof(int) * 4);
    memcpy(params.weight_dims, weightsDims, sizeof(int) * 4);
    memcpy(params.output_dims, outputsDims, sizeof(int) * 4);
    params.input_dims[3] += inputsPad;
    params.weight_dims[3] += weightsPad;
    params.output_dims[3] += outputPad;
    params.stride = stride;
    memcpy(params.input_halo_pad, inputHaloPad, sizeof(int) * 4);
    params.ifmap_start = ifmapStart;
    params.kern_start = kernStart;
    params.accum_results = accumulate;
    params.read_inputs = readInputs;
    params.read_weights = readWeights;
    params.send_results = sendResults;
    // The systolic array kernel in gem5 uses the same
    // activation type/params structures.
    memcpy(&params.act_type, &(actInfo->function), sizeof(activation_type));
    memcpy(&params.act_params, &(actInfo->params), sizeof(activation_param_t));
    return std::unique_ptr<volatile int>(
            invokeSystolicArrayAndReturn(accelId, params));
#else
    return nullptr;
#endif
}*/

void SmvDepthwiseConvolutionOp::tile() {
    // This function will tile (if necessary) the input/weight/output tensors
    // of the convolution operator into smaller tensor tiles so that each tile
    // can fit in the corresponding scratchpad of the accelerator.
    // TODO: A lot of networks have back to back convolutional layers, it would
    // be much more efficient not to retile in between them. That can be
    // achieved by directly sending the output tiles to the next convolutional
    // layer instead of merging them into a single output tensor first. It's
    // sort of operator fusing that two back-to-back convolution operators are
    // tiled only once.
    tiledTensors = smaug::smv::dwconv::TilingOptimizer::doTiling(this);
}

void SmvDepthwiseConvolutionOp::run() {
    auto input = getInput(Inputs);
    auto kernels = getInput(Kernels);
    auto output = getOutput(Outputs);
    const TensorShape& inputShape = input->getShape();
    const TensorShape& kernelShape = kernels->getShape();
    const TensorShape& outputShape = output->getShape();
    assert(inputShape.getLayout() == DataLayout::NHWC);
    assert(kernelShape.getLayout() == DataLayout::NHWC);
    assert(outputShape.getLayout() == DataLayout::NHWC);
    dout(2) << *kernels << "\n";

    /*{
        //auto stats = gem5::ScopedStats(
          //      stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
        tiledTensors[1].copyDataToAllTiles();
    }*/

    runNHWC(tiledTensors[0], tiledTensors[1], tiledTensors[2]);

    {
       // auto stats = gem5::ScopedStats(
         //       stats::kTensorFinalStart, stats::kTensorFinalEnd);
        tiledTensors[2].untile();
    }
}

}  // namespace smaug
