import torch
from glob import glob
ckpts = glob('./outputs/base_detection/**/*.ckpt', recursive=True)
for ckpt in ckpts:
    print('keep weight only', ckpt)
    st = torch.load(ckpt)['state_dict']
    torch.save(st, ckpt)
