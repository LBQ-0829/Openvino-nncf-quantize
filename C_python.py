import sys
import os
import cv2
import torch
import numpy as np
from openvino.runtime import Core


def model_proc(img, qp_in):
    """
    :param img: 10bit 输入
    img.shape  = ndarray

    """
    batch_size = img.shape[0]
    img = np.array(img)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pwd = os.getcwd()
    # Load model weights.
    if qp_in >= 50:
        model_name = 'Y52_1920x1080_int8.xml'  # todo 测试不同QP只需修改此处的模型文件加载路径
        # print('I_Y_QP52.pth')
    elif 44 <= qp_in < 50:
        model_name = 'I_Y_QP47.pth'
        # print('I_Y_QP47.pth')
    elif 37 <= qp_in < 44:
        model_name = 'I_Y_QP40.pth'
    else:
        model_name = 'I_Y_QP33.pth'
        # print('qp33')
    model_path =os.path.join(pwd, model_name)
    ie = Core()
    model = ie.read_model(model=model_path)
    model = ie.compile_model(model=model, device_name='CPU')
    output_layer = model.output(0)

    if not bool(model):
        print("Dictionary is empty")

    lq_tensor = torch.from_numpy(img) / 1023.0     # todo bit depth
    lq_tensor = lq_tensor.to(device).unsqueeze(1)
    # lq_tensor = lq_tensor.half()
    qp_tensor = torch.from_numpy(np.array(qp_in))/100.0
    qp_tensor = qp_tensor.to(device).unsqueeze(0)
    qp_tensor = qp_tensor.repeat(batch_size)

    with torch.no_grad():
        hq_out = model([qp_tensor, lq_tensor])[output_layer]
        # hq_tensor = model(qp_tensor, lq_tensor)
        hq_out = hq_out.squeeze(1)
        hq_out = np.clip(hq_out, 0.0, 1.0)
        hq_out = hq_out * 1023         # todo bit depth

        hq_out = hq_out.astype(np.uint16)  #   直接截断吗？
        hq_out = hq_out.tolist()
    return hq_out


img = cv2.imread("C:/Users/16860/Desktop/ILF-CNN/test_pic.bmp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_y = img[:, :, 0]
img_y = np.expand_dims(img_y, 0)
qp = 52
out = model_proc(img_y, qp)


