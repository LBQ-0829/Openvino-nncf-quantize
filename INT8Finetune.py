import torch
import nncf
from nncf import NNCFConfig

image_size = 224
OUTPUT_DIR = './'

# 配置NNCF的参数来指定压缩
nncf_config_dict = {
    "input_info": {"sample_size": [1, 3, image_size, image_size]},
    "log_dir": str(OUTPUT_DIR),  # The log directory for NNCF-specific logging outputs.
    "compression": {
        "algorithm": "quantization",  # Specify the algorithm here.
    },
}
nncf_config = NNCFConfig.from_dict(nncf_config_dict)

# DataLoader
nncf_config = register_default_init_args(nncf_config, train_loader)

# 创建一个nncf使用的模型对象
compression_ctrl, model = create_compressed_model(model, nncf_config)




compression_lr = init_lr / 10
optimizer = torch.optim.Adam(model.parameters(), lr=compression_lr)

# Train for one epoch with NNCF.
train(train_loader, model, criterion, optimizer, epoch=0)

# Evaluate on validation set after Quantization-Aware Training (QAT case).
acc1_int8 = validate(val_loader, model, criterion)

print(f"Accuracy of tuned INT8 model: {acc1_int8:.3f}")
print(f"Accuracy drop of tuned INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8:.3f}")


# save part
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
checkpoint = {
    'state_dict': compressed_model.state_dict(),
    'compression_state': compression_ctrl.get_compression_state(),
    ...
}
torch.save(checkpoint, path)

# load part
resuming_checkpoint = torch.load(path)
compression_state = resuming_checkpoint['compression_state']
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config, compression_state=compression_state)
state_dict = resuming_checkpoint['state_dict']

# load model in a preferable way
    load_state(compressed_model, state_dict, is_resume=True)
    # or when execution mode on loading is the same as on saving:
    # save and load in a single GPU mode or save and load in the (Distributed)DataParallel one, not in a mixed way
    compressed_model.load_state_dict(state_dict)
