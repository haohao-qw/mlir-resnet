import torch
import onnx
import numpy as np
from onnx import numpy_helper


import mlir.dialects.Tiara as Tiara
import mlir
from mlir.dialects.func import *
from mlir.dialects.EX import *
from mlir.dialects import func
from mlir.ir import *
# import torchvision.models as models

# model = models.resnet18(weights=True)
# model.eval()

# torch.onnx.export(model,               # model being run
#                     torch.randn(1, 3, 224, 224), # model input (or a tuple for multiple inputs)
#                     "resnet18.onnx",   # where to save the model (can be a file or file-like object)
#                     export_params=True,        # store the trained parameter weights inside the model file
#                     opset_version=11,          # the ONNX version to export the model to
#                     do_constant_folding=True,  # whether to execute constant folding for optimization
#                     input_names = ['input'],   # the model's input names
#                     output_names = ['output'], # the model's output names
#                     dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                     'output' : {0 : 'batch_size'}})


# 导入模型，统计模型中有哪些节点类型
model = onnx.load("/root/workspace/mlir-ex/python/Onnx/resnet18.onnx")

node_map={}
node_nexts={}

recode=[]
   
input_nodes=[]
for i in model.graph.input:
    input_nodes.append(i.name)
    node_map[i.name]=None

for n in model.graph.node:
    if input_nodes.count(n.name)>0:
        recode.append(n.name)
    node_map[n.name]=n
    node_nexts[n.name]=n.output

node_nexts[input_nodes[0]]=["/conv1/Conv"]

weight_bias={}
for initializer in model.graph.initializer:
    W = numpy_helper.to_array(initializer)
    weight_bias[initializer.name]=W
    

def get_node_attr(onnx_node: onnx.NodeProto):
    attr_map = {}
    for attr in onnx_node.attribute:
        attribute_name = attr.name
        if attr.type == onnx.AttributeProto.INTS:
            attribute_value = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            attribute_value = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            attribute_value = list(attr.strings)
        elif attr.type == onnx.AttributeProto.INT:
            attribute_value = attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            attribute_value = attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            attribute_value = attr.s
        else:
            attribute_value = None
        attr_map[attribute_name] = attribute_value
    return attr_map

node_que=[]
node_visited=set()

node_que=input_nodes

while len(node_que)>0:
    cur_node=node_que.pop(0)
    if cur_node in node_visited:
        continue
    node_visited.add(cur_node)
    for n in node_nexts[cur_node]:
        if n not in node_visited:
            node_que.append(n)
    if cur_node is None:
        continue
    cur_node_obj=node_map[cur_node]
