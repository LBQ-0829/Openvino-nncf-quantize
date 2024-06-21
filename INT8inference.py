from openvino.runtime import Core
import cv2 as cv
import numpy as np
from dataset_img import FrameYUV, get_w_h, write_YUV420_Y, write_YUV420
import os
import math


def calculate_psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


ie = Core()

model_xml = './weights/Y_fp32.xml'
model = ie.read_model(model=model_xml)
compiled_model = ie.compile_model(model=model, device_name='CPU')

output_layer = compiled_model.output(0)

# read YUV
raw_yuv_path = 'D:/dataset/DIV2K/yuv/valid/GT/0854_2040x1064.yuv'
com_yuv_path = 'D:/dataset/DIV2K/yuv/valid/QP37/0854_2040x1064.yuv'
yuv_raw = open(raw_yuv_path, 'rb')
yuv_compress = open(com_yuv_path, 'rb')
wh = get_w_h(raw_yuv_path)
yuv_raw_YUV = FrameYUV.read_YUV420(yuv_raw, wh[0], wh[1])
yuv_cmp_YUV = FrameYUV.read_YUV420(yuv_compress, wh[0], wh[1])

yuv_raw_y = yuv_raw_YUV._Y
yuv_raw_u = yuv_raw_YUV._U
yuv_raw_v = yuv_raw_YUV._V
raw_save_path = './raw.yuv'

yuv_com_y = yuv_cmp_YUV._Y
yuv_com_u = yuv_cmp_YUV._U
yuv_com_v = yuv_cmp_YUV._V
com_save_path = './com.yuv'

yuv_com_y = np.expand_dims(yuv_com_y, 0).astype(np.float32) / 255.0
yuv_com_y = np.expand_dims(yuv_com_y, 0)

qp = np.array(37) / 100.0
qp = np.expand_dims(qp, 0).astype(np.float32)

result_out = compiled_model([qp, yuv_com_y])[output_layer]

result_out = np.squeeze(np.squeeze(result_out))
result_out = np.clip(result_out, 0.0, 1.0)
result_out = result_out * 255
result_out = result_out.astype(np.uint8)
enhance_psnr = calculate_psnr(result_out, yuv_raw_y)
unenhance_psnr = calculate_psnr(yuv_cmp_YUV._Y, yuv_raw_y)
print(f'unEnhance PSNR : {unenhance_psnr}\n Enhance PSNR : {enhance_psnr}')

# write_YUV420(com_save_path, result_out, yuv_com_u, yuv_com_v)

