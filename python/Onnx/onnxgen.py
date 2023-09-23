import onnx
import onnxruntime
import numpy as np
from onnx import helper
from onnx import TensorProto

# 首先构造输入输出节点
input_node = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,3,32,32])
weight=np.random.randn(3,3,3,3).astype(np.float32)
bias=np.random.randn(3).astype(np.float32)
w1=helper.make_tensor('w1', TensorProto.FLOAT, [3,3,3,3], weight)
b1=helper.make_tensor('b1', TensorProto.FLOAT, [3], bias)
w2=helper.make_tensor('w2', TensorProto.FLOAT, [3,3,3,3], weight)
b2=helper.make_tensor('b2', TensorProto.FLOAT, [3], bias)
output_node = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,3,32,32])

# 构造onnx节点
Conv1_node=helper.make_node('Conv', inputs=['input', 'w1', 'b1'], outputs=['Conv1'],name='Conv1',kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1],dilations=[1,1])
Relu1_node=helper.make_node('Relu', inputs=['Conv1'], outputs=['Relu1'],name='Relu1')
Conv2_node=helper.make_node('Conv', inputs=['Relu1', 'w2', 'b2'], outputs=['Conv2'],name='Conv2', kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1],dilations=[1,1])
Relu2_node=helper.make_node('Relu', inputs=['Conv2'], outputs=['Relu2'],name='Relu2')
Add_node=helper.make_node('Add', inputs=['Relu2','Conv1'], outputs=['output'],name='Add')

# 构造图
graph=helper.make_graph([Conv1_node,Relu1_node,Conv2_node,Relu2_node,Add_node], 'gen', [input_node], [output_node],initializer=[w1,b1,w2,b2])
# 构造onnx模型
model_=helper.make_model(graph,producer_name='onnx-example')

# 检测模型是否符合定义
onnx.checker.check_model(model_)
onnx.save(model_, 'gen.onnx')
# 加载模型
model_ = onnx.load('gen.onnx')
# print(model_)

# 推理
ort_session = onnxruntime.InferenceSession('gen.onnx')
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
ort_inputs = {input_name: dummy_input}
ort_outs = ort_session.run([output_name], ort_inputs)
print(ort_outs)
