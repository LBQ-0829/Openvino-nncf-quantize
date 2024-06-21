import torch
from ILF_Net import Generator

device = 'cpu'
generator = Generator(1, 48).to(device)

weights = torch.load('./I_Y_QP52_optimize_6382.pth', map_location='cpu')

# keys_to_remove = [key for key in weights.keys() if 'esa' in key]
#
# for key in keys_to_remove:
#     del weights[key]

generator.load_state_dict(weights)
# model.eval()
model_outpath = 'I_Y_QP52_optimize_.pth'
torch.save(generator, model_outpath)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in generator.parameters())/1e6))
print("Model Transform Finish!")