#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np
import sys 
#sys.path.insert(0, '/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug/')
import smaug as sg

# number of classes to classify = 1000
num_classes = 1000;

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float16)

def inverted_residual_block(input_tensor, config_list, stage,block):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names

  Returns:
    Output tensor for the block.
  """
  multiplier = config_list[0] 
  kernel_size = config_list[1]
  s = config_list[2]
  output_channels = config_list[3]
  add = config_list[4]
  chkpt = config_list[5]
  input_channel_size = input_tensor.dims(3)
  input_row =  input_tensor.dims(1)
  input_col = input_tensor.dims(2)
  expansion_size = multiplier * input_channel_size # assuming NHWC layout for input 
  contraction_size = output_channels 
  conv_name_base = 'conv' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block

  # Tensors
  
  conv0_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data(
          (expansion_size, 1, 1, input_channel_size)))
  bn0_add = sg.Tensor(data_layout=sg.NHWC,tensor_data = generate_random_data((1,input_row,input_col,expansion_size)))
  
  conv1_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (1, kernel_size, kernel_size, expansion_size)))
  bn1_add = sg.Tensor(data_layout= sg.NHWC, tensor_data=generate_random_data((1,input_row//s,input_col//s,expansion_size)))
  
  conv2_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (contraction_size, 1, 1, expansion_size)))
  bn2_add = sg.Tensor( data_layout=sg.NHWC, tensor_data=generate_random_data((1,input_row//s,input_col//s,contraction_size)))
  
  if ( multiplier != 1):
    x = sg.nn.convolution(
        input_tensor, conv0_tensor, stride=[1, 1], padding="valid",
        name='exp_' + conv_name_base )
    x = sg.math.add(x,bn0_add, name='exp_'+ bn_name_base)
    x = sg.nn.relu(x, name = "bn0_relu")
    '''x = sg.nn.batch_norm(
        x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
        activation="relu", name='exp_'+ bn_name_base )'''
    
    x = sg.nn.depth_wise_convolution(
        x, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  else :
    x = sg.nn.depth_wise_convolution(
        input_tensor, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  x= sg.math.add(x,bn1_add,name = 'depth_wise_'+bn_name_base)
  x = sg.nn.relu(x, name ="dw_relu" + bn_name_base)
  '''
  x = sg.nn.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation="relu", name= 'depth_wise' + bn_name_base)'''
  x = sg.nn.convolution(
      x, conv2_tensor, stride=[1, 1], padding="valid",
      name='contr_'+conv_name_base )
  x=sg.math.add(x,bn2_add,name='contr_'+bn_name_base)
  
  '''
  x = sg.nn.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name='contr_'+bn_name_base)'''
  if add == True:
    x = sg.nn.relu(x,"contr_relu" + bn_name_base) 
    if chkpt: 
      x = sg.math.add(x, input_tensor, name=add_name + "_chkpt")
    else:
      x = sg.math.add(x, input_tensor, name=add_name )

  else:
    if chkpt:
      x = sg.nn.relu(x,"contr_relu" + bn_name_base + "_chkpt")
    else:
      x = sg.nn.relu(x,"contr_relu" + bn_name_base)

  return x


def create_mosaic():
   
  with sg.Graph(name="mosaic_smv", backend="SMV") as graph:

    #IBN_layer_config_list = [[3, 3, 2, 32, False, False], [2, 3, 1, 32, True, False], [5, 5, 2, 64, False, False], [3, 3, 1, 64, True, True], [2, 3, 1, 64, True, False], [3, 3, 1, 64 ,True, False], [6, 5, 2 ,128, False, False], [3, 3, 1, 128, True, True], [3, 3, 1, 128, True, False], 
    #                         [3, 3,  1, 128, True, False], [6, 3, 1, 160, False, False], [4, 3, 1, 160, True, True], [6, 3, 1, 192, False, False], [2, 5, 1, 96, False, False], [4, 5, 1, 96, True, False], [4, 5, 1, 96, True, True]]
    IBN_layer_config_list = [[3, 3, 2, 32, False, False], [5, 5, 2, 64, False, False], [3, 3, 1, 64, True, True],  [6, 5, 2 ,128, False, False], [3, 3, 1, 128, True, True],  
                              [6, 3, 1, 160, False, False], [4, 3, 1, 160, True, True], [6, 3, 1, 192, False, False], [2, 5, 1, 96, False, False] ]
 
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 512, 512, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 3)))
    
    bn0_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,256,256,32)))
    
    x = sg.input_data(input_tensor, name="input")
    x = sg.nn.convolution(
        x, conv0_tensor, stride=[2, 2], padding="same", name="conv0")
    x = sg.math.add(x,bn0_add_tensor,name='bn0_add_tensor')
    x = sg.nn.relu(x, "bn0_relu")
    
    for stage in range(len(IBN_layer_config_list)):
      x = inverted_residual_block(x, IBN_layer_config_list[stage], stage , 'a' )

    conv8_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((448,1,1,96)))
    bn8_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32, 32, 448)))    
    x = sg.nn.convolution(x,conv8_tensor, stride = [1,1], padding="same",name = "conv8")
    x = sg.math.add(x,bn8_add_tensor,name = "bn8_add_tensor")
    x = sg.nn.relu(x, " bn8_relu")
     
    x1 = sg.nn.avg_pool(x, pool_size=[1, 1], stride=[1, 1], name="avg_pool_1")
    x2 = sg.nn.avg_pool(x, pool_size=[1, 1], stride=[1, 1], name="avg_pool_2")
    x3 = sg.nn.avg_pool(x, pool_size=[1, 1], stride=[1, 1], name="avg_pool_3")

    s1a, s1b = sg.python.ops.array_ops.split(x1, 2, 3, "split_1")
    s2a, s2b = sg.python.ops.array_ops.split( x2, 2, 3, "split_2")
    s3a, s3b = sg.python.ops.array_ops.split(x3, 2, 3, "split_3")

    dw_conv_1a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,5,5,224)))
    dw_conv_1b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,3,3,224)))
    bias_dw_1a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))
    bias_dw_1b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))

    dw_conv_2a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,5,5,224)))
    dw_conv_2b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,3,3,224)))
    bias_dw_2a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))
    bias_dw_2b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))

    dw_conv_3a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,5,5,224)))
    dw_conv_3b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((1,3,3,224)))
    bias_dw_3a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))
    bias_dw_3b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,224)))
    pt_conv_1a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1,1,224)))
    pt_conv_1b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1, 1,224)))
    bias_pt_1a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))
    bias_pt_1b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))

    pt_conv_2a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1,1,224)))
    pt_conv_2b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1, 1,224)))
    bias_pt_2a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))
    bias_pt_2b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))

    pt_conv_3a = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1,1,224)))
    pt_conv_3b = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((64,1, 1,224)))
    bias_pt_3a = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))
    bias_pt_3b = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,32,32,64)))

    dw_s1a = sg.nn.depth_wise_convolution(s1a, dw_conv_1a, [1,1], "same")
    bias_dw_s1a = sg.math.add(dw_s1a, bias_dw_1a, " dw_add_s1a")
    bias_dw_relu_s1a = sg.nn.relu(bias_dw_s1a, " s1a_dw_relu")
    pt_s1a = sg.nn.convolution(bias_dw_relu_s1a, pt_conv_1a, [1,1],"same")
    pt_bias_s1a = sg.math.add(pt_s1a, bias_pt_1a, " s1a_pt_bias")
    pt_relu_s1a = sg.nn.relu(pt_bias_s1a, " s1a_pt_relu")

    dw_s1b = sg.nn.depth_wise_convolution(s1b, dw_conv_1b, [1,1], "same")
    bias_dw_s1b = sg.math.add(dw_s1b, bias_dw_1b, " dw_add_s1b")
    bias_dw_relu_s1b = sg.nn.relu(bias_dw_s1b, " s1b_dw_relu")
    pt_s1b = sg.nn.convolution(bias_dw_relu_s1b, pt_conv_1b, [1,1],"same")
    pt_bias_s1b = sg.math.add(pt_s1b, bias_pt_1b, " s1b_pt_bias")
    pt_relu_s1b = sg.nn.relu(pt_bias_s1b, " s1b_pt_relu")

    dw_s2a = sg.nn.depth_wise_convolution(s2a, dw_conv_2a, [1,1], "same")
    bias_dw_s2a = sg.math.add(dw_s2a, bias_dw_2a, " dw_add_s2a")
    bias_dw_relu_s2a = sg.nn.relu(bias_dw_s2a, " s2a_dw_relu")
    pt_s2a = sg.nn.convolution(bias_dw_relu_s2a, pt_conv_2a, [1,1],"same")
    pt_bias_s2a = sg.math.add(pt_s2a, bias_pt_2a, " s2a_pt_bias")
    pt_relu_s2a = sg.nn.relu(pt_bias_s2a, " s2a_pt_relu_chkpt")

    dw_s2b = sg.nn.depth_wise_convolution(s2b, dw_conv_2b, [1,1], "same")
    bias_dw_s2b = sg.math.add(dw_s2b, bias_dw_2b, " dw_add_s2b")
    bias_dw_relu_s2b = sg.nn.relu(bias_dw_s2b, " s2b_dw_relu")
    pt_s2b = sg.nn.convolution(bias_dw_relu_s2b, pt_conv_2b, [1,1],"same")
    pt_bias_s2b = sg.math.add(pt_s2b, bias_pt_2b, " s2b_pt_bias")
    pt_relu_s2b = sg.nn.relu(pt_bias_s2b, " s2b_pt_relu")
    
    dw_s3a = sg.nn.depth_wise_convolution(s3a, dw_conv_3a, [1,1], "same")
    bias_dw_s3a = sg.math.add(dw_s3a, bias_dw_3a, " dw_add_s3a")
    bias_dw_relu_s3a = sg.nn.relu(bias_dw_s3a, " s3a_dw_relu")
    pt_s3a = sg.nn.convolution(bias_dw_relu_s3a, pt_conv_3a, [1,1],"same")
    pt_bias_s3a = sg.math.add(pt_s3a, bias_pt_3a, " s3a_pt_bias")
    pt_relu_s3a = sg.nn.relu(pt_bias_s3a, " s3a_pt_relu")

    dw_s3b = sg.nn.depth_wise_convolution(s3b, dw_conv_3b, [1,1], "same")
    bias_dw_s3b = sg.math.add(dw_s3b, bias_dw_3b, " dw_add_s3b")
    bias_dw_relu_s3b = sg.nn.relu(bias_dw_s3b, " s3b_dw_relu")
    pt_s3b = sg.nn.convolution(bias_dw_relu_s3b, pt_conv_3b, [1,1],"same")
    pt_bias_s3b = sg.math.add(pt_s3b, bias_pt_3b, " s3b_pt_bias")
    pt_relu_s3b = sg.nn.relu(pt_bias_s3b, " s3b_pt_relu_chkpt")

    concat_s1 = sg.python.ops.array_ops.concat([pt_relu_s1a, pt_relu_s1b], 3, " concat_s1_chkpt")
    concat_s2 = sg.python.ops.array_ops.concat([pt_relu_s2a, pt_relu_s2b], 3, " concat_s2")
    concat_s3 = sg.python.ops.array_ops.concat([pt_relu_s3a, pt_relu_s3b], 3, " concat_s3")

    concat_all = sg.python.ops.array_ops.concat( [x, concat_s1, concat_s2, concat_s3], 3, " concat_all")

    pt_conv_concat = sg.Tensor(data_layout=sg. NHWC, tensor_data=generate_random_data((128,1,1,832)))
    x = sg.nn.convolution(concat_all, pt_conv_concat, [1,1], "same", name=" final_pt_conv")




    """conv10_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((num_classes, 1, 1, 1280)))
    bn10_add_tensor = sg.Tensor(data_layout = sg.NHWC, tensor_data=generate_random_data((1,1,1,num_classes)))
    
    x = sg.nn.convolution(x,conv10_tensor, stride = [1,1], padding="valid",name = "conv10")
    x=sg.math.add(x,bn10_add_tensor,name="bn10_add_tensor")
    x = sg.nn.relu(x,"bn10_rleu")"""

    
    return graph

if __name__ != "main":
  mosaic_model = create_mosaic()
  mosaic_model.print_summary()
  mosaic_model.write_graph()
