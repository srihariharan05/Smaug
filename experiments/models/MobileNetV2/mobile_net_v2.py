#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np
import sys 
sys.path.insert(0, '/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug/')
import smaug as sg

# number of classes to classify = 1000
num_classes = 1000;

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float16)

def inverted_residual_block(input_tensor, multiplier, output_channels,s, add, stage,block):
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
  kernel_size = 3
  input_channel_size = input_tensor.dims(3)
  expansion_size = multiplier * input_channel_size # assuming NHWC layout for input 
  contraction_size = output_channels 
  conv_name_base = 'conv' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

  # Tensors
  
  conv0_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data(
          (expansion_size, 1, 1, input_channel_size)))
  bn0_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1,expansion_size )))
  bn0_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  bn0_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  bn0_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  conv1_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (1, kernel_size, kernel_size, expansion_size)))
  bn1_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  bn1_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  bn1_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  bn1_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, expansion_size)))
  conv2_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (contraction_size, 1, 1, expansion_size)))
  bn2_mean_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, contraction_size)))
  bn2_var_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, contraction_size)))
  bn2_gamma_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, contraction_size)))
  bn2_beta_tensor = sg.Tensor(
      data_layout=sg.NC, tensor_data=generate_random_data((1, contraction_size)))

  if ( multiplier != 1):
    x = sg.nn.convolution(
        input_tensor, conv0_tensor, stride=[1, 1], padding="valid",
        name='exp_' + conv_name_base )
    x = sg.nn.batch_norm(
        x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
        activation="relu", name='exp_'+ bn_name_base )
    x = sg.nn.depth_wise_convolution(
        x, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  else :
    x = sg.nn.depth_wise_convolution(
        input_tensor, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  x = sg.nn.batch_norm(
      x, bn1_mean_tensor, bn1_var_tensor, bn1_gamma_tensor, bn1_beta_tensor,
      activation="relu", name= 'depth_wise' + bn_name_base)
  x = sg.nn.convolution(
      x, conv2_tensor, stride=[1, 1], padding="valid",
      name='contr_'+conv_name_base )
  x = sg.nn.batch_norm(
      x, bn2_mean_tensor, bn2_var_tensor, bn2_gamma_tensor, bn2_beta_tensor,
      name='contr_'+bn_name_base)
  if add == True:  
    x = sg.math.add(x, input_tensor, name=add_name)

  return x


def create_mobilenetV2():
   
  with sg.Graph(name="mobilenetV2_smv", backend="SMV") as graph:
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 224, 224, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 3)))
    bn0_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 32)))
    bn0_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 32)))
    bn0_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 32)))
    bn0_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 32)))
    fc_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((10, 7 * 7 * 2048)))

    x = sg.input_data(input_tensor, name="input")
    x = sg.nn.convolution(
        x, conv0_tensor, stride=[2, 2], padding="same", name="conv0")
    x = sg.nn.batch_norm(
        x, bn0_mean_tensor, bn0_var_tensor, bn0_gamma_tensor, bn0_beta_tensor,
        activation="relu", name="bn0")
    #stage 1 
    x= inverted_residual_block(x, 1, 16, 1, False, 1, '1')
    #stage 2
    x = inverted_residual_block(x, 6, 24, 2, False, 2, '1')
    x = inverted_residual_block(x, 6, 24, 1, True, 2,"2")
    #stage 3
    x = inverted_residual_block(x, 6, 32, 2, False, 3, '1')
    x = inverted_residual_block(x, 6, 32, 1, True,3, '2')
    x = inverted_residual_block(x, 6, 32, 1, True, 3, '3')
    #stage 4 
    x = inverted_residual_block(x, 6, 64, 2, False, 4, '1')
    x = inverted_residual_block(x, 6, 64, 1, True, 4, '2')
    x = inverted_residual_block(x, 6, 64, 1, True, 4, '3')
    x = inverted_residual_block(x, 6, 64, 1, True, 4, '4')
    #stage 5 
    x = inverted_residual_block(x, 6, 96, 1, False, 5, '1')
    x = inverted_residual_block(x, 6, 96, 1, True, 5, '2')
    x = inverted_residual_block(x, 6, 96, 1, True, 5, '3')
    #stage 6
    x = inverted_residual_block(x, 6, 160, 2, False, 6, '1')
    x = inverted_residual_block(x, 6, 160, 1, True, 6, '2')
    x = inverted_residual_block(x, 6, 160, 1, True, 6, '3')
    #stage 7
    x = inverted_residual_block(x, 6, 320, 1, False, 7, '1')

    conv8_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1280, 1, 1, 320)))
    bn8_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 1280)))
    bn8_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 1280)))
    bn8_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 1280)))
    bn8_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, 1280)))
    
    x = sg.nn.convolution(x,conv8_tensor, stride = [1,1], padding="valid",name = "conv8")
    x = sg.nn.batch_norm(
        x, bn8_mean_tensor, bn8_var_tensor, bn8_gamma_tensor, bn8_beta_tensor,
        activation="relu", name="bn8")
     
    x = sg.nn.avg_pool(x, pool_size=[7, 7], stride=[1, 1], name="avg_pool")

    conv10_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((num_classes, 1, 1, 1280)))
    bn10_mean_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, num_classes)))
    bn10_var_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, num_classes)))
    bn10_gamma_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, num_classes)))
    bn10_beta_tensor = sg.Tensor(
        data_layout=sg.NC, tensor_data=generate_random_data((1, num_classes)))
    
    x = sg.nn.convolution(x,conv10_tensor, stride = [1,1], padding="valid",name = "conv10")
    x = sg.nn.batch_norm(
        x, bn10_mean_tensor, bn10_var_tensor, bn10_gamma_tensor, bn10_beta_tensor,
        activation="relu", name="bn10")

    
    return graph

if __name__ != "main":
  mobilenetV2 = create_mobilenetV2()
  mobilenetV2.print_summary()
  mobilenetV2.write_graph()
