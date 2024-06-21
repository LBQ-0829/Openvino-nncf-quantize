from openvino.runtime import Core, serialize
import nncf
from dataset_img import BaseDataset_img

core = Core()
ov_model = core.read_model('./dynamic_qp52.xml')

img_dir = 'E:/Download_from_edge/DIV2K/DIV2K_valid_HR/'
val_loader = BaseDataset_img(img_dir, 1, 52)
calibration_dataset = nncf.Dataset(val_loader)  # todo 这里的图片输入数据大小时候会影响模型的精度

quantized_model = nncf.quantize(ov_model, calibration_dataset, preset=nncf.QuantizationPreset.MIXED)
nncf_int8_path = './dynamic_qp52_INT8.xml'
serialize(quantized_model, nncf_int8_path)
