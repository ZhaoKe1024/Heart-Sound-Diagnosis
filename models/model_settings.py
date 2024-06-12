#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/12 10:20
# @Author: ZhaoKe
# @File : model_settings.py
# @Software: PyCharm
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2
from pretrained.wav2vec import Wav2Vec
from models.classifiers import LSTM_Classifier, LSTM_Attn_Classifier

def get_model(use_model, device):
    model = None
    if use_model == "mnv2":
        model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
        input_size = (1, 87, 128)
    elif use_model == "wav2v+lstm_attn":
        wav2v = Wav2Vec(pretrained=True, pretrained_path="../pretrained/wav2vec_large.pt", device=device)
        model = nn.Sequential(
            wav2v,
            LSTM_Attn_Classifier(512, 64, 5, return_attn_weights=True, attn_type="dot")
        )
        input_size = (1, 147000)
    elif use_model == "lstm_attn":
        model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=5,
                                     return_attn_weights=True, attn_type="dot").to(device)
        input_size = (87, 128)
    return model
