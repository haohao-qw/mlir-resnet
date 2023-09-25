import torch
import onnx
import numpy as np
from onnx import numpy_helper


import mlir.dialects.EX as EX
import mlir
from mlir.dialects.func import *
from mlir.dialects.EX import *
from mlir.dialects import func
from mlir.ir import *
from mlir.dialects import builtin


import torchvision.models as models
model = models.resnet18(weights=True)
model.eval()

torch.onnx.export(model,               # model being run
                    torch.randn(1, 3, 224, 224), # model input (or a tuple for multiple inputs)
                    "resnet18.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})


class ModuleOp:
    def __init__(self):
        self.ctx = mlir.ir.Context()
        self.ctx.allow_unregistered_dialects = True
        self.loc=Location.name("test",context=self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.module=builtin.ModuleOp()
        self.weight_map = {}
        self.input_map = {}
        self.output_map={}
        self.name_node = {}
        self.queue_node=[]
        self.model=None
        self.op_map={}
        
    def init(self, file_path,input_name):
        self.queue_node.append(input_name)
        self.model = onnx.load(file_path)
        for init in self.model.graph.initializer:
            self.weight_map[init.name] = numpy_helper.to_array(init)
        for key in self.weight_map:
            self.weight_map[key]=self.build_constant_op(DenseElementsAttr.get(self.weight_map[key]),locname=key)
        for node in self.model.graph.node:
            self.name_node[node.name] = node
            for input in node.input:
                if input not in self.input_map:
                    self.input_map[input] = []
                self.input_map[input].append(node.name)
            for output in node.output:
                if output not in self.output_map:
                    self.output_map[output] = []
                self.output_map[output].append(node.name)
    
    def __del__(self):
        try:
            self.loc.__exit__(None, None, None)
        except:
            pass
        try:
            self.ctx.__exit__(None, None, None)
        except:
            pass

    def get_loc(self, names):
        if isinstance(names, str):
            loc= Location.fused([Location.name(names)], context=self.ctx)
            return loc
        elif isinstance(names, list):
            loc= Location.fused([Location.name(n) for n in names], context=self.ctx)
            return loc
        else:
            raise RuntimeError("Unknown names:{}".format(names))

    def build_return_op(self, values, locname:str):
        func.ReturnOp(values, loc=self.get_loc(locname))
    def build_constant_op(self, value, locname:str):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.ConstantOp(output, value,loc=self.get_loc(locname))
    def build_conv_op(self,input, weight, bias, stride, kernel_shape, padding, dilations, locname:str,group=None, auto_pad=None):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.ConvOp(output, input, weight, bias, stride, kernel_shape, padding, dilations,loc=self.get_loc(locname))
    def build_relu_op(self,input, locname:str):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.ReluOp(output, input, loc=self.get_loc(locname))
    def build_maxpool_op(self, input, dilations, kernel_shape, pads, strides, locname, auto_pad=None, ceil_mode=None, p=None, storage_order=None):
        output=UnrankedTensorType.get(F64Type.get())
        results=EX.MaxPoolOp(output,output, input, dilations, kernel_shape, pads, strides, loc=self.get_loc(locname)).results
        return results
    def build_globalaveragepool_op(self,input, locname:str):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.GlobalAveragePoolOp(output, input, loc=self.get_loc(locname))
    def build_add_op(self,input1, input2, locname:str):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.AddOp(output, input1, input2, loc=self.get_loc(locname))
    def build_flatten_op(self,input, locname:str):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.FlattenOp(output, input, loc=self.get_loc(locname))
    def build_gemm_op(self, A, B, C, locname, alpha=None, beta=None, transA=None, transB=None):
        output=UnrankedTensorType.get(F64Type.get())
        return EX.GemmOp(output, A, B, C,loc=self.get_loc(locname))

    def dfs(self,node_name):
        if node_name in self.op_map:
            return self.op_map[node_name]
        node=self.name_node[node_name]
        op_type=node.op_type
        if op_type=="Conv":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            weight_name=node.input[1]
            weight=self.weight_map[weight_name]
            bias_name=node.input[2]
            bias=self.weight_map[bias_name]
            dilations=node.attribute[0].ints
            dilations= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in dilations])
            kernel_shape=node.attribute[2].ints
            kernel_shape= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in kernel_shape])
            pads=node.attribute[3].ints
            pads= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in pads])
            strides=node.attribute[4].ints
            strides= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in strides])
            op=self.build_conv_op(pre_op,weight,bias,strides,kernel_shape,pads,dilations,locname=node_name)
            self.op_map[node_name]=op
            return op
        elif op_type=="Relu":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            op=self.build_relu_op(pre_op,locname=node_name)
            self.op_map[node_name]=op
            return op
        elif op_type=="MaxPool":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            dilations=[1,1]
            dilations=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in dilations])
            kernel_shape=node.attribute[1].ints
            kernel_shape=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in kernel_shape])
            pads=node.attribute[2].ints
            pads=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in pads])
            strides=node.attribute[3].ints
            strides=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in strides])
            op=self.build_maxpool_op(pre_op,dilations,kernel_shape,pads,strides,locname=node_name)
            self.op_map[node_name]=op
            return op
        elif op_type=="GlobalAveragePool":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            op=self.build_globalaveragepool_op(pre_op,locname=node_name)
            self.op_map[node_name]=op
            return op
        elif op_type=="Add":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op_a=self.dfs(pre_name)
            input_name=node.input[1]
            pre_name=self.output_map[input_name][0]
            pre_op_b=self.dfs(pre_name)
            op=self.build_add_op(pre_op_a,pre_op_b,locname=node_name)
            self.op_map[node_name]=op
            return op
        elif op_type=="Flatten":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            op=self.build_flatten_op(pre_op,locname=node_name).result
            self.op_map[node_name]=op
            return op
        elif op_type=="Gemm":
            input_name=node.input[0]
            pre_name=self.output_map[input_name][0]
            pre_op=self.dfs(pre_name)
            weight_name=node.input[1]
            weight=self.weight_map[weight_name]
            bias_name=node.input[2]
            bias=self.weight_map[bias_name]
            op=self.build_gemm_op(pre_op,weight,bias,locname=node_name)
            self.op_map[node_name]=op
            return op
        else:
            raise RuntimeError("Unknown op_type:{}".format(op_type))

    def Begin(self):
        with InsertionPoint(self.module.body):
            input_type=RankedTensorType.get((2,3), F64Type.get())
            output_type=RankedTensorType.get((2,3), F64Type.get())
            functype=FunctionType.get([input_type],[output_type])
            funcop=func.FuncOp("module_1", functype,loc=self.get_loc("module_1"))
            with InsertionPoint(funcop.add_entry_block()):
                array = np.random.random((2,3)).astype(np.float64)
                attr = DenseElementsAttr.get(array)
                input=self.build_constant_op(attr,locname="input")
                self.op_map["input"]=input
                self.output_map["input"]=["input"]
                while len(self.queue_node) != 0:
                    cur_name = self.queue_node.pop(0)
                    node = self.name_node[cur_name]
                    op_type=node.op_type
                    if op_type=="Conv":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op=self.op_map[pre_name]
                        weight_name=node.input[1]
                        weight=self.weight_map[weight_name]
                        bias_name=node.input[2]
                        bias=self.weight_map[bias_name]
                        dilations=node.attribute[0].ints
                        dilations= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in dilations])
                        kernel_shape=node.attribute[2].ints
                        kernel_shape= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in kernel_shape])
                        pads=node.attribute[3].ints
                        pads= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in pads])
                        strides=node.attribute[4].ints
                        strides= ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in strides])
                        op=self.build_conv_op(pre_op,weight,bias,strides,kernel_shape,pads,dilations,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="Relu":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue                       
                        pre_op=self.op_map[pre_name]
                        op=self.build_relu_op(pre_op,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="MaxPool":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op=self.op_map[pre_name]
                        dilations=[1,1]
                        dilations=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in dilations])
                        kernel_shape=node.attribute[1].ints
                        kernel_shape=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in kernel_shape])
                        pads=node.attribute[2].ints
                        pads=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in pads])
                        strides=node.attribute[3].ints
                        strides=ArrayAttr.get([IntegerAttr.get(IntegerType.get_signless(64),item) for item in strides])
                        op=self.build_maxpool_op(pre_op,dilations,kernel_shape,pads,strides,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="GlobalAveragePool":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op=self.op_map[pre_name]
                        op=self.build_globalaveragepool_op(pre_op,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="Add":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op_a=self.op_map[pre_name]
                        input_name=node.input[1]
                        pre_name=self.output_map[input_name][0]
                        pre_op_b=self.op_map[pre_name]
                        op=self.build_add_op(pre_op_a,pre_op_b,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="Flatten":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op=self.op_map[pre_name]
                        op=self.build_flatten_op(pre_op,locname=cur_name)
                        self.op_map[cur_name]=op
                    elif op_type=="Gemm":
                        input_name=node.input[0]
                        pre_name=self.output_map[input_name][0]
                        if pre_name not in self.op_map:
                            self.queue_node.append(cur_name)
                            continue
                        pre_op=self.op_map[pre_name]
                        weight_name=node.input[1]
                        weight=self.weight_map[weight_name]
                        bias_name=node.input[2]
                        bias=self.weight_map[bias_name]
                        op=self.build_gemm_op(pre_op,weight,bias,locname=cur_name)
                        self.op_map[cur_name]=op
                    output_list=node.output
                    node_list=[]    
                    for output_name in output_list:
                        if output_name =="output":
                            break
                        next_node=self.input_map[output_name]
                        # 问题在这里，如果有多个next_node，就会出现问题 TODO
                        # node_list.append(next_node[0])
                        for name in next_node:
                            node_list.append(name)
                    for node_name in node_list:
                        self.queue_node.append(node_name)
                # func.ReturnOp([v0.result],loc=self.get_loc("module_3"))
        self.module.print(enable_debug_info=True)

    def DfsBegin(self,file_path,input_name,output_name):
        with InsertionPoint(self.module.body):
            input_type=UnrankedTensorType.get(F64Type.get())
            output_type=UnrankedTensorType.get(F64Type.get())
            functype=FunctionType.get([input_type],[output_type])
            funcop=func.FuncOp("module_1", functype,loc=self.get_loc("begin"))
            with InsertionPoint(funcop.add_entry_block()):
                self.init(file_path,input_name)
                array = np.random.random((2,3)).astype(np.float64)
                attr = DenseElementsAttr.get(array)
                input=self.build_constant_op(attr,locname="input")
                self.op_map["input"]=input
                self.output_map["input"]=["input"]
                self.dfs(output_name)
                self.build_return_op([self.op_map[output_name]],locname="end")
        self.module.print(enable_debug_info=True)

if __name__=="__main__":
    model=ModuleOp()
    model.DfsBegin("resnet18.onnx","/conv1/Conv","/fc/Gemm")