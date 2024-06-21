import openvino.runtime as ov
import torch
from openvino.tools import mo
from dataset_img import BaseDataset_img
import nncf
from openvino.tools.pot import save_model
from ILF_Net import Generator

img_dir = 'E:/Download_from_edge/DIV2K/DIV2K_valid_HR/'
QP_list = [52]
for input_QP in QP_list:
    model = torch.load('./I_Y_QP' + f'{input_QP}' + '_optimize.pth')
    out_dir = model.export(format="openvino", dynamic=True, half=False)
    model.eval()

    val_loader = BaseDataset_img(img_dir, 1, input_QP)

    calibration_dataset = nncf.Dataset(val_loader)  # todo 这里的图片输入数据大小时候会影响模型的精度
    quantized_model = nncf.quantize(model, calibration_dataset, preset=nncf.QuantizationPreset.PERFORMANCE)  # 量化为INT8  对模型进行量化：此函数会对模型应用 8 位量化
    # 最大限度地降低精度影响，但模型精度可能会出现轻微下降

    ov_quantized_model = mo.convert_model(quantized_model.cpu(), input_shape=[[4], [4, 1, 2040, 1352]])

    int8_ir_path = f"weights/Y{input_QP}_4x1x2040x1352.xml"
    ov.serialize(ov_quantized_model, int8_ir_path)  # 对 OpenVINO™ 工具套件 IR 模型进行序列化
    print(f"Save INT8 model: {int8_ir_path}")
