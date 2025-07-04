#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np
import sys 
#sys.path.insert(0, '/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug/')
import smaug as sg

# number of classes to classify = 1000

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float16)

def edsr_residual_block(input_tensor, layer1, layer2, stage):
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
  input_channel_size = input_tensor.dims(3)
  input_row =  input_tensor.dims(1)
  input_col = input_tensor.dims(2)
  
  conv_name_base = 'conv' + str(stage)
  bn_name_base = 'bn' + str(stage)
  add_name = 'add' + str(stage)
  relu_name = 'relu' + str(stage)

  # Tensors
  
  conv0_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data(
          (layer1[0], layer1[1], layer1[2], input_channel_size)))
  bn0_add = sg.Tensor(data_layout=sg.NHWC,tensor_data = generate_random_data((1,input_row,input_col,layer1[0])))
  
  conv1_tensor = sg.Tensor(
      data_layout=sg.NHWC, tensor_data=generate_random_data(
          (layer2[0], layer2[1], layer2[2], layer1[0])))
  bn1_add = sg.Tensor(data_layout= sg.NHWC, tensor_data=generate_random_data((1,input_row,input_col,layer2[0])))
  
  x = sg.nn.convolution(
        input_tensor, conv0_tensor, stride=[1, 1], padding="same",
        name=conv_name_base + "_1")
  x = sg.math.add(x,bn0_add, name=bn_name_base + " _1")
  x = sg.nn.relu(x, name = relu_name + " _1")
  
  x = sg.nn.convolution(
        x, conv1_tensor, stride=[1, 1], padding="same",
        name=conv_name_base + "_2")
  x = sg.math.add(x,bn1_add, name=bn_name_base + " _2")
  x = sg.math.add(x, input_tensor, name=add_name)


  return x


def create_edsr():
   
  with sg.Graph(name="edsr_smv", backend="SMV") as graph:
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 48, 48, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 3)))
    
    bn0_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,48,48,32)))

    conv6_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 32)))
    bn6_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1, 48, 48, 32)))

    conv7_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((12, 3, 3, 32)))
    bn7_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1, 48, 48, 12)))

    input_conv = sg.input_data(input_tensor, name="input")
    input_conv = sg.nn.convolution(
        input_conv, conv0_tensor, stride=[1, 1], padding="same", name="conv0")
    input_conv = sg.math.add(input_conv,bn0_add_tensor,name='bn0_add_tensor')
    
    #stage1
    x = edsr_residual_block(input_conv,[32,3,3,32], [32,3,3,32], 1)
    x = edsr_residual_block(x,[32,3,3,32], [32,3,3,32], 1)
    x = edsr_residual_block(x,[32,3,3,32], [32,3,3,32], 1)
    x = edsr_residual_block(x,[32,3,3,32], [32,3,3,32], 1)
    x = edsr_residual_block(x,[32,3,3,32], [32,3,3,32], 1)

    x = sg.nn.convolution(
        x, conv6_tensor, stride=[1, 1], padding="same", name="conv6_chkpt")
    x = sg.math.add(x,bn6_add_tensor,name='bn6_add_tensor')
    x = sg.math.add(x, input_conv, " add_inp_conv")

    x = sg.nn.convolution(
        x, conv7_tensor, stride=[1, 1], padding="same", name="conv7")
    x = sg.math.add(x,bn7_add_tensor,name='bn7_add_tensor')

    x = sg.nn.relu(x, " final_relu")
    
    return graph

if __name__ != "main":
  edsr = create_edsr()
  edsr.print_summary()
  edsr.write_graph()
