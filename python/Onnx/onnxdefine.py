import onnx
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        y = self.relu(x)
        y = self.conv2(y)
        y = self.relu(y)
        # add算子
        out = torch.add(x, y)
        return out
# 随机权重初始化 
net = Net()
for name, param in net.named_parameters():
    if 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.01)

# torch推理
dummy_input = torch.randn(1, 3, 32, 32)
output = net(dummy_input)

# 保存onnx模型
torch.onnx.export(net, dummy_input, "net.onnx", verbose=True,input_names=['input'],output_names=['output'])
print("save onnx model done.")

# onnx推理
model=onnx.load("net.onnx")
onnx.checker.check_model(model)
ort_session = onnxruntime.InferenceSession("net.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name
dummy_input = dummy_input.numpy()
ort_inputs = {input_name: dummy_input}
ort_outs = ort_session.run([output_name], ort_inputs)

output = output.detach().numpy()
print("output shape:",output.shape)
print("ort_outs shape:",ort_outs[0].shape)

point=5
total=0
num=0
for a,b in zip(output,ort_outs[0]):
    for c,d in zip(a,b):
        for e,f in zip(c,d):
            for i,j in zip(e,f):
                total+=1
                if round(i,point)==round(j,point):
                    num+=1
print("val:",num/total)