#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/8 9:55
# @Author: ZhaoKe
# @File : audioprocess.py
# @Software: PyCharm
import numpy as np
import librosa
import torch
import torchaudio


def read_wav_mel(wavpath: str, length: int = 44100):
    sig, sr = librosa.load(path=wavpath)
    if len(sig) > length:
        st = (len(sig) - length) // 2
        sig = sig[st:st + length]
        return torch.from_numpy(sig).to(torch.float32)
    elif len(sig) < length:
        # print(len(sig), wavpath)
        tmp = np.zeros(length)
        st = (length - len(sig)) // 2
        tmp[st:st + len(sig)] = sig
        return torch.from_numpy(tmp).to(torch.float32)
    else:
        return torch.from_numpy(sig).to(torch.float32)


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))
