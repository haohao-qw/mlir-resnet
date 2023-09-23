import onnx
from onnx import helper
from onnx import numpy_helper
import sys,getopt

model_name="resnet18.onnx"

#加载模型
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

model=loadOnnxModel(model_name)

#获取节点和节点的输入输出名列表，一般节点的输入将来自于上一层的输出放在列表前面，参数放在列表后面
def getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node,input_name,output_name

#获取对应输入信息
def getInputTensorValueInfo(input_name,model):
    in_tvi = []
    for name in input_name:
        for params_input in model.graph.input:
            if params_input.name == name:
               in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

#获取对应输出信息
def getOutputTensorValueInfo(output_name,model):
    out_tvi = []
    for name in output_name:
        out_tvi = [inner_output for inner_output in model.graph.value_info if inner_output.name == name]
        if name == model.graph.output[0].name:
            out_tvi.append(model.graph.output[0])
    return out_tvi


#获取对应超参数值
def getInitTensorValue(input_name,model):
    init_t = []
    for name in input_name:
        init_t = [init for init in model.graph.initializer if init.name == name]
    return init_t

#获取节点数量
def getNodeNum(model):
    return len(model.graph.node)


#获取节点类型
def getNodetype(model):
    op_name = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type not in op_name:
            op_name.append(model.graph.node[i].op_type)
    return op_name

#获取节点名列表
def getNodeNameList(model):
    NodeNameList = []
    for i in range(len(model.graph.node)):
        NodeNameList.append(model.graph.node[i].name)
    return NodeNameList

#获取模型的输入信息
def getModelInputInfo(model):
    return model.graph.input[0]

#获取模型的输出信息
def getModelOutputInfo(model):
    return model.graph.output[0:4]

# 获取模型节点权重或者bias信息,以numpy数组形式返回
def getModelInitInfo(model,op_name):
    init_t = []
    for i in range(len(model.graph.initializer)):
        if model.graph.initializer[i].name == op_name:
            init_t.append(numpy_helper.to_array(model.graph.initializer[i]))
    return init_t

# 获取节点属性信息
def getNodeAttribute(model,op_name):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == op_name:
            return model.graph.node[i].attribute

# 遍历模型的所有节点，获取每个节点的信息
def getAllNodeInfo(model):
    ret=[]
    for i in range(len(model.graph.node)):
        ret.append(model.graph.node[i])
    return ret