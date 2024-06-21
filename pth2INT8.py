import openvino.runtime as ov
import torch
from openvino.tools import mo
from dataset_img import BaseDataset_img
# from ILF_Net import Generator
import nncf
from openvino.tools.pot import save_model

# model = Generator(in_channel=1, nf=48)
# weights = torch.load('./I_Y_QP40.pth', map_location='cpu')  # TODO cpu or cuda:0
# model.load_state_dict(weights)
img_dir = 'E:/Download_from_edge/DIV2K/DIV2K_valid_HR/'

QP_list = [52]
for input_QP in QP_list:
    model = torch.load('./utils/I_Y_QP' + f'{input_QP}' + '_optimize_.pth')
    model.eval()

    val_loader = BaseDataset_img(img_dir, 1, input_QP)

    calibration_dataset = nncf.Dataset(val_loader)  # todo 这里的图片输入数据大小时候会影响模型的精度
    quantized_model = nncf.quantize(model, calibration_dataset, preset=nncf.QuantizationPreset.MIXED)  # 量化为INT8

    # ov_model = mo.convert_model(model.cpu(), input_shape=[[1], [1, 1, 1080, 1920]])
    ov_quantized_model = mo.convert_model(quantized_model.cpu(), input_shape=[[8], [8, 1, 1352, 2040]])

    # fp32_ir_path = f"weights/Y_fp32.xml"
    # ov.serialize(ov_model, fp32_ir_path)
    # print(f"Save FP32 model: {fp32_ir_path}")

    int8_ir_path = f"weights/Y{input_QP}_8x1x160x160_int8.xml"
    ov.serialize(ov_quantized_model, int8_ir_path)
    print(f"Save INT8 model: {int8_ir_path}")
