#include <stdbool.h>
#include <stdio.h>

#include "smaug/operators/common.h"
#include "smaug/operators/smv/kernels/params.h"
#include "smaug/operators/smv/kernels/load_store_fp16_data.h"
#include "smaug/operators/smv/kernels/activation_functions_simd.h"


#ifdef __cplusplus
extern "C" {
#endif

/** \ingroup AladdinKernels
 *
 * Perform a 3D convolution with one kernel on an image, with reduction in NHWC
 * format. This is the vectorized implementation.
 *
 * @param host_inputs Host inputs buffer in NHWC.
 * @param host_weights Host weights buffer in NHWC.
 * @param host_results Host results buffer in NHWC.
 * @param inputs Local inputs buffer in NHWC.
 * @param weights Local weights buffer in NHWC.
 * @param results Local results buffer in NHWC.
 * @param inputs_dims Dimensions of the inputs.
 * @param weights_dims Dimensions of the weights.
 * @param results_dims Dimensions of the results.
 * @param inputs_align_pad Alignment padding size on the channel dimension of
 *        the inputs.
 * @param weights_pad Alignment padding size on the channel dimension of the
 *        weights.
 * @param results_pad Alignment padding size on the channel dimension of the
 *        results.
 * @param inputs_halo_pad Padding sizes on top, bottom, left and right of the
 * input 2D feature maps.
 * @param row_stride Stride size on the row dimension.
 * @param col_stride Stride size on the col dimension.
 * @param ifmap_start If the input contains more channels than the weights,
 *        start from this one. Otherwise this should always be zero.
 * @param kern_start If the weights contain more kernels than the results buffer can
 *        fit, start from this one. Otherwise this should always be zero.
 * @param accumulate If the original weight tensor is tiled channelwise, this
 *        should be set to true in order to avoid resetting the result buffer
 *        for non-first weight tiles.
 * @param read_inputs Load inputs from the host. Set to false if the input
 *        activations can be reused from the last invocation.
 * @param read_weights Load weights from the host. Set to false if the weights
 *        can be reused from the last invocation.
 * @param send_results Send the results to the host memory if this is true.
 * @param act_function Activation function the operator runs.
 * @param act_params Parameters for the activation function.
 * @param sampling Simulation samplng settings.
 */
void smv_depthwise_conv3d_nhwc_vec_fxp(float16* host_inputs,
                             float16* host_weights,
                             float16* host_results,
                             float* inputs,
                             float* weights,
                             float* results,
                             int inputs_dims[4],
                             int weights_dims[4],
                             int results_dims[4],
                             int inputs_align_pad,
                             int weights_pad,
                             int results_pad,
                             int inputs_halo_pad[4],
                             int row_stride,
                             int col_stride,
                             int ifmap_start,
                             int kern_start,
                             bool accumulate,
                             bool read_inputs,
                             bool read_weights,
                             bool send_results,
                             activation_type act_function,
                             activation_param_t act_params,
                             SamplingInfo* sampling) {
    int result_rows = results_dims[1];
    int result_cols = results_dims[2];
    int result_height = results_dims[3];
    int results_size = results_dims[0] * result_rows * result_cols *
                       (result_height + results_pad);

    int k_rows = weights_dims[1];
    int k_cols = weights_dims[2];
    int k_height = weights_dims[3];
    int k_pad = weights_pad;
    int weights_size = weights_dims[0] * k_rows * k_cols * (k_height + k_pad);

    int a_rows = inputs_dims[1];
    int a_cols = inputs_dims[2];
    int a_height = inputs_dims[3];
    int a_pad = inputs_align_pad;
    int inputs_size = inputs_dims[0] * a_rows * a_cols * (a_height + a_pad);

    int top_pad = inputs_halo_pad[0];
    int bottom_pad = inputs_halo_pad[1];
    int left_pad = inputs_halo_pad[2];
    int right_pad = inputs_halo_pad[3];
    int end_row = a_rows + top_pad + bottom_pad - k_rows + 1;
    int end_col = a_cols + left_pad + right_pad - k_cols + 1;

    int input_row_with_pad = a_rows + top_pad +bottom_pad;
    int input_col_with_pad = a_cols + left_pad + right_pad;
    int valid_row_end = a_rows - 1;
    int valid_col_end = a_cols - 1;

    int in_row, in_col;

    assert (((result_height + results_pad) <= (k_height + k_pad)) && "results channel should always be less than weights " );
    
    volatile v8fp_t results_buffer [NUM_PE_INSTS][NUM_MACC_INSTS];
    volatile v8fp_t act_reg [NUM_PE_INSTS][NUM_MACC_INSTS];
    //v8fp_t dw_conv_reg[NUM_PE_INSTS][NUM_MACC_INSTS];
    // Results in NHWC.
    #if (ACC_256_GOPS == 1)
    // Kernels and input are in NHWC.
    /*VEC_ARRAY_3D(v8fp_t, _kernel, weights, k_cols, k_height + k_pad);
    VEC_ARRAY_3D(v8fp_t, _a, inputs, a_cols, a_height + a_pad);
    VEC_ARRAY_3D(
            v8fp_t, _result, results, result_cols, result_height + results_pad);*/
    ARRAY_3D(float, _kernel, weights, k_cols, k_height + k_pad);
    ARRAY_3D(float, _a, inputs, a_cols, a_height + a_pad);
    ARRAY_3D(
            float, _result, results, result_cols, result_height + results_pad);
    volatile v8fp_t kernel_reg = { 0,0,0,0,0,0,0,0};
    #elif (ACC_1_TOPS == 1)
    // Kernels and input are in NHWC.
    VEC_ARRAY_3D(v32fp_t, _kernels, weights, k_cols, (k_height + k_pad)/RESULT_ALIGNMENT);
    VEC_ARRAY_3D(v32fp_t, _a, inputs, a_cols, (a_height + a_pad)/RESULT_ALIGNMENT);
    VEC_ARRAY_3D(
            v32fp_t, _result, results, result_cols, (result_height + results_pad)/RESULT_ALIGNMENT);
    volatile v32fp_t kernel_reg; // = { 0,0,0,0,0,0,0,0};
    #elif ( ACC_32_TOPS ==1)
    // Kernels and input are in NHWC.
    VEC_ARRAY_3D(v256fp_t, _kernels, weights, k_cols, (k_height + k_pad)/RESULT_ALIGNMENT);
    VEC_ARRAY_3D(v256fp_t, _a, inputs, a_cols, (a_height + a_pad)/RESULT_ALIGNMENT);
    VEC_ARRAY_3D(
            v256fp_t, _result, results, result_cols, (result_height + results_pad)/RESULT_ALIGNMENT);
    volatile v256fp_t kernel_reg; 
    #endif
    int num_chan_blocks = (result_height +results_pad) / NUM_PE_INSTS;
    int num_row_blocks = FRAC_CEIL(input_row_with_pad, (NUM_MACC_INSTS * row_stride));
    int num_col_blocks = FRAC_CEIL(input_col_with_pad,(VECTOR_SIZE * col_stride)); 
    // Number of effective kernels for this invocation. The weights can contain
    // more kernels than the results buffer can fit the output feature maps,
    // where the number of effective kernels will be the number of feature maps
    // in the results.

    // Load inputs and weights if needed.
    if (read_inputs)
        host_load_fp16(inputs, host_inputs, inputs_size, 0, 0);
    if (read_weights)
        host_load_fp16(weights, host_weights, weights_size, 0, 0);

    // Set up the sample sizes and factors.
    //int pe_block_sample = num_chan_blocks; // + 1;
    int kern_row_sample = k_rows;
    int kern_col_sample = k_cols;
    int chan_block_sample = num_chan_blocks;// + 1;
    int sample_num = sampling->num_sample_iterations;
    int row_block_sample = num_row_blocks;
    int col_block_sample = num_col_blocks;
    if (sampling->level >= Low)
        chan_block_sample = min2(chan_block_sample, sample_num);
    if (sampling->level >= Medium) {
        kern_row_sample = min2(kern_row_sample, sample_num);
        kern_col_sample = min2(kern_col_sample, sample_num);
    }
    if (sampling->level >= High)
        row_block_sample = min2(row_block_sample, sample_num);
    if (sampling->level >= VeryHigh) {
        col_block_sample = min2(col_block_sample, sample_num);
    }

    setSamplingFactor("ofmap_block_iteration",
                      (num_chan_blocks + 1) * 1.0 / chan_block_sample);
    setSamplingFactor("k_row", k_rows * 1.0 / kern_row_sample);
    setSamplingFactor("k_col", k_cols * 1.0 / kern_col_sample);
    setSamplingFactor(
            "macc_iteration", (num_row_blocks + 1) * 1.0 / row_block_sample);
    setSamplingFactor("vector_iteration",
                      num_col_blocks * 1.0 / col_block_sample);
    

    bool new_kern = false;
    int num_eff_maccs =0, num_eff_vectors =0;
    ofmap_block_iteration:
    for (int ofmap_iters = 0; ofmap_iters < chan_block_sample;
         ofmap_iters++) {  // Result channel blocks
        int ofmap_offset = ofmap_iters * NUM_PE_INSTS;
        // If we have less than eight output channels, don't run the extra ones.
        int kEffNumPeInsts = min2(result_height - ofmap_offset, NUM_PE_INSTS);
        // Kernel rows
        pe_loop:
        for ( int pe_iter =0; pe_iter < kEffNumPeInsts; pe_iter ++ ){
            k_row:
            for (int kern_row = 0; kern_row < kern_row_sample; kern_row++) {
                new_kern = true;
                k_col:
                for (int kern_col = 0; kern_col < kern_col_sample;
                    kern_col++) {  // Kernel cols
                    // This loops over all the input channels in groups of
                    // VECTOR_SIZE * NUM_MACC_INSTS.
                    kernel_reg[pe_iter] = (float) _kernel[kern_row][kern_col][kern_start + ofmap_offset + pe_iter];
                    macc_iteration:
                    for (int ifmap_row_iters = 0; ifmap_row_iters < row_block_sample;
                        ifmap_row_iters++) {
                            int ifmap_row_offset = ifmap_row_iters * NUM_MACC_INSTS * row_stride;
                            num_eff_maccs = min2(NUM_MACC_INSTS, (end_row /*+ kern_row*/ - ifmap_row_offset)/row_stride );
                            vector_iteration:
                                for ( int ifmap_col_iters =0; ifmap_col_iters < col_block_sample; ifmap_col_iters ++){
                                    int ifmap_col_offset = ifmap_col_iters * VECTOR_SIZE * col_stride;
                                    num_eff_vectors = min2( VECTOR_SIZE, (end_col /*+ kern_col*/ - ifmap_col_offset)/col_stride);
                                    macc_loop:
                                    for ( int macc_iter =0; macc_iter < num_eff_maccs; macc_iter++){
                                        vector_loop:
                                        for ( int vector_iter =0; vector_iter < num_eff_vectors; vector_iter ++){
                                            results_buffer[pe_iter][macc_iter][vector_iter] = _result[ifmap_row_offset + macc_iter][ifmap_col_offset + vector_iter][ofmap_offset + pe_iter];
                                            in_row = ifmap_row_offset + macc_iter - top_pad + kern_row ;
                                            in_col = ifmap_col_offset + vector_iter - left_pad + kern_col; 
                                            bool in_padding = (((in_row < 0) || (in_row > valid_row_end)) || ((in_col < 0 )|| (in_col > valid_col_end)));
                                            if ( in_padding){
                                                act_reg[pe_iter][macc_iter][vector_iter] = 0;
                                            }
                                            else 
                                                act_reg[pe_iter][macc_iter][vector_iter] = _a[ifmap_row_offset + (macc_iter*row_stride) + kern_row][ifmap_col_offset + (vector_iter*col_stride) + kern_col][ifmap_start + ofmap_offset + pe_iter];
                                            if ( new_kern){
                                                new_kern = false;
                                                results_buffer[pe_iter][macc_iter][vector_iter] = kernel_reg[pe_iter]* act_reg[pe_iter][macc_iter][vector_iter];

                                            }
                                            else 
                                                results_buffer[pe_iter][macc_iter][vector_iter] += kernel_reg[pe_iter]* act_reg[pe_iter][macc_iter][vector_iter];
                                            _result[ifmap_row_offset + macc_iter][ifmap_col_offset + vector_iter][ofmap_offset + pe_iter] =  results_buffer[pe_iter][macc_iter][vector_iter];
                                        }
                                    }
                                }
                        }
                    }
                }
            }
    }
    // Only run activation functions when the results are finished.
    if (act_function != NO_ACTIVATION && send_results) {
        activation_fun_vec(
                results, results, results_size, act_function, act_params);
    }
    // Store results to the host memory if needed.
    if (send_results)
        host_store_fp16(results, host_results, results_size, 0, 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

