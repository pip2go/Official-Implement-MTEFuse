import sys
from utils import *
import warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 降低内存碎片[3,4](@ref)
warnings.filterwarnings("ignore")
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from PIL import Image
import numpy as np
import os
import torch
from tqdm import tqdm
import time
import imageio
import torchvision.transforms as transforms
from model import MTEFuse_model as net
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')

model = net(in_channel=1)
model_path = ""
use_gpu = False

if use_gpu:

    model = model.cuda()
    model.cuda()
    model.load_state_dict(torch.load(model_path))

else:

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)


def fusion(Data_vi, Data_ir):
    data_vi = list(Path(Data_vi).iterdir())
    data_ir = list(Path(Data_ir).iterdir())

    for num, item in enumerate(tqdm(data_vi)):

        path1 = str(data_vi[num])
        path2 = str(data_ir[num])
        img1 = Image.open(path1).convert('L')
        img2 = Image.open(path2).convert('L')

        img1_org = img1
        img2_org = img2
        tran = transforms.ToTensor()
        img1_org = tran(img1_org)
        img2_org = tran(img2_org)
        if use_gpu:
            img1_org = img1_org.cuda()
            img2_org = img2_org.cuda()
        else:
            img1_org = img1_org
            img2_org = img2_org
        img1_org = img1_org.unsqueeze(0)
        img2_org = img2_org.unsqueeze(0)

        model.eval()
        out = model(img1_org, img2_org)
        d = np.squeeze(out.detach().cpu().numpy())
        result = (d * 255).astype(np.uint8)
        Path('YDTR').mkdir(parents=True, exist_ok=True)
        imageio.imwrite('./YDTR/{:0>2}.png'.format(num + 1), result)


if __name__ == '__main__':
    Data_vi = r""
    Data_ir = r""
    fusion(Data_vi, Data_ir)
