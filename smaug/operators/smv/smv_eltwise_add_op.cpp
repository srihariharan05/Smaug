#include "smaug/core/backend.h"
#include "smaug/core/tensor_utils.h"
#include "smaug/operators/common.h"
#include "smaug/operators/smv/smv_eltwise_add_op.h"
#include "smaug/operators/smv/smv_unary_op_common.h"
#include "smaug/operators/smv/smv_kernels.h"
#include "smaug/utility/debug_stream.h"

namespace smaug {

// The tile dispatcher for elementwise addition.
void SmvEltwiseAddOp::runX(TiledTensor& inputs0,
                           TiledTensor& inputs1,
                           TiledTensor& outputs) {
    assert(inputs0.size() == inputs1.size() &&
           inputs0.size() == outputs.size());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs0", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_inputs1", getInputsMemType());
    setArrayMemTypeIfSimulating(
            smv::kEltwiseOpHw, "host_results", getOutputsMemType());
    for (int i = 0; i < inputs0.size(); i++) {
        dout(1) << "Input0: " << i << ", input1: " << i << ", output: " << i
                << "\n";
        //tiledTensors[0].getTile(i)->tensor->allocateStorage<float16>();
        //tiledTensors[1].getTile(i)->tensor->allocateStorage<float16>();
        inputs0[i]->allocateStorage(Float16);
        inputs1[i]->allocateStorage(Float16);
        Tensor* input0Tile = inputs0.getTileWithData(i);
        Tensor* input1Tile = inputs1.getTileWithData(i);
        Tensor* outputTile = outputs[i];

        outputTile->allocateStorage(Float16);
        const TensorShape& inputShape = input0Tile->getShape();
        const TensorShape& outputShape = outputTile->getShape();

        
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs0",
                        input0Tile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_inputs1",
                        input1Tile->data<float16>(),
                        inputShape.storageSize() * sizeof(float16));
        mapArrayToAccel(smv::kEltwiseOpHw, "host_results",
                        outputTile->data<float16>(),
                        outputShape.storageSize() * sizeof(float16));
        flush_memory((void*)outputTile->data<float16>(),outputShape.storageSize() * sizeof(float16));
        flush_memory((void*)input0Tile->data<float16>(),inputShape.storageSize() * sizeof(float16));
        flush_memory((void*)input1Tile->data<float16>(),inputShape.storageSize() * sizeof(float16));
        invokeKernel(smv::kEltwiseOpHw, smv_eltwise_add_nc_vec_fxp,
                     input0Tile->data<float16>(), input1Tile->data<float16>(),
                     outputTile->data<float16>(), smv::spad0, smv::spad1,
                     smv::spad2, inputShape.storageSize());
    }
}

void SmvEltwiseAddOp::tile() {
    // We reuse the unary op tiler for the elementwise addition operator.
    using namespace smaug::smv::unary;
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    int maxTileSize =
            std::min(SmvBackend::SpadSize() / inputs0->getDataTypeSize(),
                     inputs0->getShape().storageSize());
    TensorShape tileShape(
            { 1, maxTileSize }, DataLayout::NC, SmvBackend::Alignment);
    tiledTensors[0] = generateTiledTensorPerBatchNC(
        inputs0, tileShape, this, false);
    tiledTensors[1] = generateTiledTensorPerBatchNC(
        inputs1, tileShape, this, false);
    tiledTensors[2] = generateTiledTensorPerBatchNC(
        outputs, tileShape, this, false);
        stat_num_of_tiles += tiledTensors[0].size() + tiledTensors[1].size() + tiledTensors[2].size();

}

void SmvEltwiseAddOp::run() {
    auto inputs0 = getInput(Input0);
    auto inputs1 = getInput(Input1);
    auto outputs = getOutput(Outputs);
    const TensorShape& inputs0Shape = inputs0->getShape();
    const TensorShape& inputs1Shape = inputs1->getShape();
    const TensorShape& outputsShape = outputs->getShape();
    assert(inputs0Shape == inputs1Shape && inputs0Shape == outputsShape);

    /*{
        //auto stats = gem5::ScopedStats(
          //      stats::kTensorPrepStart, stats::kTensorPrepEnd);
        tiledTensors[0].copyDataToAllTiles();
        tiledTensors[1].copyDataToAllTiles();
    }*/

    runX(tiledTensors[0], tiledTensors[1], tiledTensors[2]);

    {
        //auto stats = gem5::ScopedStats(
          //      stats::kTensorFinalStart, stats::kTensorFinalEnd);
        flattenTiledTensor(tiledTensors[2], outputs);
    }
}

}  // namespace smaug
