from openvino.runtime import Core
import cv2 as cv
import numpy as np
from dataset_img import FrameYUV, get_w_h, write_YUV420_Y, write_YUV420
import os
import math
# from ILF_net import Generator
import torch


def calculate_psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# model = Generator(in_channel=1, nf=48)
# weights = torch.load('./I_Y_QP52_best.pth', map_location='cpu')  # TODO cpu or cuda:0
# model.load_state_dict(weights)
model = torch.load('./I_Y_QP40_optimize.pth')
model.eval()

# lq_tensor = torch.from_numpy(np.array(img))/255.0     # todo bit depth
# lq_tensor = lq_tensor.to(device).unsqueeze(0).unsqueeze(0)
# qp_tensor = torch.from_numpy(np.array(qp_in))/100.0
# qp_tensor = qp_tensor.to(device).unsqueeze(0)
# with torch.no_grad():
#     hq_tensor = model(qp_tensor, lq_tensor)
#     hq_result = hq_tensor.cpu().squeeze(0).squeeze(0)
#     hq_result = hq_result.numpy()
#     hq_result = np.clip(hq_result, 0.0, 1.0)
#     hq_arr = hq_result * 255         # todo bit depth
#
#     hq_arr = hq_arr.astype(np.uint8)

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

# write_y = open(save_path, 'rb')
# write_YUV420(raw_save_path, yuv_raw_y,yuv_raw_u,yuv_raw_v)

lq_tensor = np.expand_dims(yuv_com_y, 0).astype(np.float32) / 255.0
lq_tensor = torch.from_numpy(np.expand_dims(lq_tensor, 0)).to('cpu')
# lq_tensor = torch.from_numpy(np.array(yuv_com_y))/255.0     # todo bit depth
# lq_tensor = lq_tensor.to('cpu').unsqueeze(0).unsqueeze(0)
print(yuv_com_y.shape)

qp = np.array(37) / 100.0
qp = torch.from_numpy(np.expand_dims(qp, 0).astype(np.float32)).to('cpu')
print(qp.shape)

with torch.no_grad():
    hq_tensor = model(qp, lq_tensor)
    hq_result = hq_tensor.cpu().squeeze(0).squeeze(0)
    hq_result = hq_result.numpy()
    hq_result = np.clip(hq_result, 0.0, 1.0)
    hq_arr = hq_result * 255         # todo bit depth

    hq_arr = hq_arr.astype(np.uint8)

enhance_psnr = calculate_psnr(hq_arr, yuv_raw_y)
unenhance_psnr = calculate_psnr(yuv_cmp_YUV._Y, yuv_raw_y)
print(f'unEnhance PSNR : {unenhance_psnr}\n Enhance PSNR : {enhance_psnr}')
# cv.imwrite('./rec_pic.png', result_out)
write_YUV420(com_save_path, hq_arr, yuv_com_u, yuv_com_v)

