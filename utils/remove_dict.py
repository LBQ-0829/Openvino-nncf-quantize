import torch


weights = torch.load(r'C:\Users\16860\Desktop\fsdownload/I_Y_QP52_optimize_6382.pth', map_location='cpu')
list_key = list(weights.keys())
keys_to_remove = [key for key in weights.keys() if 'esa' in key or 'gama_line' in key or 'beta_line' in key]

for key in keys_to_remove:
    del weights[key]
torch.save(weights, './I_Y_QP52_optimize_6382.pth')

changed_dict = torch.load('./I_Y_QP52_optimize_6382.pth')
for key in changed_dict.keys():
    print(key)