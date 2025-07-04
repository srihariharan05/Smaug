#!/usr/bin/env python

"""Example for creating the ResNet-50 network."""

import numpy as np
import sys 
#sys.path.insert(0, '/home/grads/s/srihariharan05/Thesis/docker_smaug/smaug/')
import smaug as sg

num_classes = 1000

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.008).astype(np.float16)

def Tucker_block(input_tensor, kernel_size, c, e, output_channel, stage):
  """ Creates a tucker block """
  input_channel = input_tensor.dims(3)
  input_rows = input_tensor.dims(1)
  input_cols = input_tensor.dims(2)
  input_num = input_tensor.dims(0)
  inp_compression = (input_channel * c) // 100
  
  out_compression = (output_channel* e) // 100 
  
  print (inp_compression); print (" ")
  print (out_compression); print ( "\n")
  inp_pt_wise_str = "Tucker_blk_" + str(stage) + "_inp_pt_wise"
  out_pt_wise_str = "Tucker_blk_" + str(stage) + "_out_pt_wise"
  reg_conv_str = "Tucker_blk_" + str(stage) + "_reg_conv"
  

  inp_pt_wise_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data = generate_random_data((inp_compression,1,1,input_channel)))
  kernel_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((out_compression,kernel_size,kernel_size,inp_compression)))
  out_pt_wise_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((output_channel,1,1,out_compression)))

  bn_inp_pt_wise = sg.Tensor( data_layout=sg.NHWC,tensor_data=generate_random_data((input_num,input_rows,input_cols,inp_compression)))
  bn_reg_conv = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((input_num,input_rows,input_cols,out_compression)))
  bn_out_pt_wise = sg.Tensor( data_layout=sg.NHWC,tensor_data=generate_random_data((input_num,input_rows,input_cols,output_channel)))
  
  x = sg.nn.convolution(
      input_tensor, inp_pt_wise_tensor, stride=[1, 1], padding="same",
      name= inp_pt_wise_str)
  x=sg.math.add(x,bn_inp_pt_wise,name=inp_pt_wise_str + "_bn")
  x = sg.nn.relu(x,name=inp_pt_wise_str + " _relu")

  x = sg.nn.convolution(
      x, kernel_tensor, stride=[1, 1], padding="same",
      name= reg_conv_str)
  x=sg.math.add(x,bn_reg_conv,name=reg_conv_str + "_bn")
  x = sg.nn.relu(x,name=reg_conv_str + " _relu")

  x = sg.nn.convolution(
      x, out_pt_wise_tensor, stride=[1, 1], padding="same",
      name= out_pt_wise_str)
  x=sg.math.add(x,bn_out_pt_wise,name=out_pt_wise_str + "_bn")
  x = sg.nn.relu(x,name=out_pt_wise_str + " _relu")

  return x

def fused_inverted_block(input_tensor,kernel_size,multiplier, output_channel,s,add,stage,block, chkpt):
  """ implements fised inverted block"""
  input_channel = input_tensor.dims(3)
  input_rows = input_tensor.dims(1)
  input_cols = input_tensor.dims(2)
  input_num = input_tensor.dims(0)
  expansion_size = multiplier * input_channel # assuming NHWC layout for input 
  contraction_size = output_channel 
  fused_conv_name_base = 'fused_blk_conv' + str(stage) + "_"+block
  out_pt_wise_str = "fused_blk_out_pt_wise" + str(stage) + "_"+block
  add_name = 'fused_blk_add' + str(stage) + "_" + block

  kernel_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((expansion_size,kernel_size,kernel_size,input_channel)))
  conv_bn_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((input_num,input_rows //s,input_cols //s,expansion_size)))
  out_pt_wise_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((output_channel,1,1,expansion_size)))
  out_pt_wise_bn_tensor = sg.Tensor(data_layout=sg.NHWC,tensor_data=generate_random_data((input_num,input_rows //s,input_cols //s,output_channel)))

  x = sg.nn.convolution(
      input_tensor, kernel_tensor, stride=[s, s], padding="same",
      name= fused_conv_name_base)
  x=sg.math.add(x,conv_bn_tensor,name=fused_conv_name_base + "_bn")
  x = sg.nn.relu(x,name=fused_conv_name_base + " _relu")

  x = sg.nn.convolution(
      x, out_pt_wise_tensor, stride=[1, 1], padding="same",
      name= out_pt_wise_str)
  x=sg.math.add(x,out_pt_wise_bn_tensor,name=out_pt_wise_str + "_bn")

  if (add):
    x = sg.nn.relu(x,name=out_pt_wise_str + " _relu")
    if chkpt:
      x= sg.math.add(input_tensor,x, name=add_name + "_chkpt" )
    else:
      x= sg.math.add(input_tensor,x, name=add_name)
  else:
    if chkpt:
      x = sg.nn.relu(x,name=out_pt_wise_str + " _relu_chkpt")
    else:
      x = sg.nn.relu(x,name=out_pt_wise_str + " _relu")

  return x



def inverted_residual_block(input_tensor, kernel_size, multiplier, output_channels,s, add, stage,block, chkpt):
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
  expansion_size = multiplier * input_channel_size # assuming NHWC layout for input 
  contraction_size = output_channels 
  conv_name_base = 'conv' + str(stage) + block
  bn_name_base = 'bn' + str(stage) + block
  add_name = 'add' + str(stage) + "_" + block
  relu_name = 'relu' + str(stage) + "_" + block

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
    
    x = sg.nn.depth_wise_convolution(
        x, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  else :
    x = sg.nn.depth_wise_convolution(
        input_tensor, conv1_tensor, stride=[s, s], padding="same",
        name='depth_wise_' + conv_name_base)
  x= sg.math.add(x,bn1_add,name = 'depth_wise_'+bn_name_base)
  x = sg.nn.relu(x, name ="dw_relu" + bn_name_base)
  
  x = sg.nn.convolution(
      x, conv2_tensor, stride=[1, 1], padding="valid",
      name='contr_'+conv_name_base )
  x=sg.math.add(x,bn2_add,name='contr_'+bn_name_base)
  
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


def create_mobiledet():
   
  with sg.Graph(name="mobiledet_smv", backend="SMV") as graph:
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    # sg.Tensors and kernels are initialized as sg.NCHW layout.
    input_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1, 224, 224, 3)))
    conv0_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((32, 3, 3, 3)))
    bn0_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,112,112,32)))
    

    x = sg.input_data(input_tensor, name="input")
    x = sg.nn.convolution(
        x, conv0_tensor, stride=[2, 2], padding="same", name="conv0")
    x= sg.math.add(x,bn0_add_tensor,name='bn0_add_tensor')
    x = sg.nn.relu(x, "bn0_relu")
    
    #stage 1 Tucker stage
    x= Tucker_block(x, 3, 25, 75, 16, 1)
    #stage 2 Fused IBN
    x = fused_inverted_block(x, 3, 8, 16, 2, False, 2, 'a',False)
    #x = fused_inverted_block(x, 3, 4, 16, 1, True, 2, 'b', False)
    #x = fused_inverted_block(x, 3, 8, 16, 1, True, 2, 'c', False)
    x = fused_inverted_block(x, 3, 4, 16, 1, True, 2, 'd', False)
    #stage 3 Fused IBN
    x = fused_inverted_block(x, 5, 8, 40, 2, False, 3, 'a', False)
    #x = fused_inverted_block(x, 3, 4, 40, 1, True, 3, 'b', False)
    #x = fused_inverted_block(x, 3, 4, 40, 1, True, 3, 'c', False)
    x = fused_inverted_block(x, 3, 4, 40, 1, True, 3, 'd', True)
    #stage 4 
    x=  inverted_residual_block(x, 3, 8, 72, 2, False, 4, 'a', False)
    #x = inverted_residual_block(x, 3, 8, 72, 1, True, 4, 'b', False)
    #x = fused_inverted_block(x, 3, 4, 72, 1, True, 4, 'c', False)
    x = fused_inverted_block(x, 3, 4, 72, 1, True, 4, 'd', False)
    #stage 5 
    x = inverted_residual_block(x, 5, 8, 96, 1, False, 5, 'a', False)
    #x = inverted_residual_block(x, 5, 8, 96, 1, True, 5, 'b', False)
    #x = inverted_residual_block(x, 3, 8, 96, 1, True, 5, 'c', False)
    #x = inverted_residual_block(x, 3, 8, 96, 1, True, 5, 'd', True)
    #stage 6 
    x = inverted_residual_block(x, 5, 8, 120, 2, False, 6, 'a', True)
    #x = inverted_residual_block(x, 3, 8, 120, 1, True, 6, 'b', False)
    #x = inverted_residual_block(x, 5, 4, 120, 1, True, 6, 'c', False)
    x = inverted_residual_block(x, 3, 8, 120, 1, True, 6, 'd', False)
    #stage 7
    x= inverted_residual_block(x, 5, 8, 384, 1, False, 7, 'a', False)
    
    conv8_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((1280, 1, 1, 384)))
    bn8_add_tensor = sg.Tensor(data_layout=sg.NHWC, tensor_data=generate_random_data((1,7,7,1280)))    
    x = sg.nn.convolution(x,conv8_tensor, stride = [1,1], padding="valid",name = "conv8")
    x = sg.math.add(x,bn8_add_tensor,name = "bn8_add_tensor")
    x = sg.nn.relu(x, " bn8_relu")
     
    x = sg.nn.avg_pool(x, pool_size=[7, 7], stride=[1, 1], name="avg_pool")

    conv10_tensor = sg.Tensor(
        data_layout=sg.NHWC, tensor_data=generate_random_data((num_classes, 1, 1, 1280)))
    bn10_add_tensor = sg.Tensor(data_layout = sg.NHWC, tensor_data=generate_random_data((1,1,1,num_classes)))
    
    x = sg.nn.convolution(x,conv10_tensor, stride = [1,1], padding="valid",name = "conv10")
    x=sg.math.add(x,bn10_add_tensor,name="bn10_add_tensor")
    x = sg.nn.relu(x,"bn10_rleu")

    return graph

if __name__ != "main":
  mobiledet = create_mobiledet()
  mobiledet.print_summary()
  mobiledet.write_graph()
