#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/16 15:12
# @Author: ZhaoKe
# @File : physionet_infer.py
# @Software: PyCharm
import sys
import torch
from models.mobilefacenet import MobileFaceNet
from datautils.audioprocess import Wave2Mel, read_wav_mel


def pretrain_setting():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = MobileFaceNet(inp_c=1, input_dim=87, latent_size=(6, 8), num_class=2, inp=1).to(device)
    cl_model.load_state_dict(torch.load(f"./ckpt/physionet/202407160841_/cls_model_780.pt"))
    cl_model.eval()
    w2m = Wave2Mel(sr=22050)
    return w2m, cl_model, device


class_names = ["Normal", "Abnormal"]
l2m = {}
m2l = {}
for i in range(len(class_names)):
    l2m[i] = class_names[i]
    m2l[class_names[i]] = i

w2m, cl_model, device = pretrain_setting()
# print(cl_model)


def test_command(input_path):
    X_mel = w2m(read_wav_mel(input_path))
    with torch.no_grad():
        X_mel = X_mel.unsqueeze(0).transpose(1, 2).to(device)  # (87, 128)
        # print("shape of X_mel:", X_mel.shape)  # torch.Size([128, 87])
        pred, _ = cl_model(torch.stack([X_mel, X_mel], dim=0), torch.tensor([0, 1], device=device))

        # print(pred)
        col = pred.argmax(-1)
        # print(col)
        res_label = None
        res = torch.inf
        for idx, val in enumerate(col):
            if pred[idx, val] < res:
                res = pred[idx, val]
                res_label = idx
        print(res_label, res)
        # v1 = pred1.argmax(-1).item()
        # v2 = pred2.argmax(-1).item()

        # res =
        # print(f"The prediction of {input_path.split('/')[-1]} is {l2m[]}")


if __name__ == '__main__':
    # test_one()
    # test_command(sys.argv[1])
    # DATA_ROOT = "F:/DATAS/heartsounds/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/"
    #
    # input_path = DATA_ROOT + "training-a/a0007.wav"
    # test_command(input_path)
    #
    # input_path = DATA_ROOT + "training-a/a0008.wav"
    # test_command(input_path)

    test_command(sys.argv[1])
