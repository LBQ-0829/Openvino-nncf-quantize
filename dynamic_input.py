import torch
import numpy as np
import onnx
from openvino.tools import mo
from openvino.runtime import serialize

model = torch.load('./utils/I_Y_QP52_optimize_.pth', map_location='cpu')
model.eval()

input_name = ['input_qp', 'input_img']
output_name = ['output']

qp = (np.array(52).astype(np.float32)) / (np.array(100).astype(np.float32))
qp = torch.from_numpy(np.array(qp)).unsqueeze(0)
x = torch.randn(1, 1, 1080, 1920, device='cpu')
input_data = (qp, x)


dynamic_axes = {'input_qp': {0: 'batch_size'},
                'input_img': {0: 'batch_size', 2: 'height', 3: 'weight'},
                'output': {0: 'batch_size', 2: 'height', 3: 'weight'}}

torch.onnx.export(model, input_data, './dynamic_qp52.onnx', input_names=input_name, output_names=output_name,
                  dynamic_axes=dynamic_axes)

ov_IR = mo.convert_model('./dynamic_qp52.onnx', compress_to_fp16=False)
serialize(ov_IR, './dynamic_qp52.xml')
