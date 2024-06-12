#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/12 10:11
# @Author: ZhaoKe
# @File : trainer_wav2v.py
# @Software: PyCharm
import torch
import torch.nn as nn
from pretrained.wav2vec import Wav2Vec
from models.classifiers import LSTM_Attn_Classifier

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
wav2v = Wav2Vec(pretrained=True, pretrained_path="../pretrained/wav2vec_large.pt", device=device)
model = nn.Sequential(
    wav2v,
    LSTM_Attn_Classifier(512, 64, 5, return_attn_weights=True, attn_type="dot")
)

print(model)
inputx = torch.randn(size=(16, 1, 147000))
output, attn = model(inputx)
print(output.shape, attn.shape)
