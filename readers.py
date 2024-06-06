#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/6 14:46
# @Author: ZhaoKe
# @File : readers.py
# @Software: PyCharm
import os
import numpy as np
import torch
from tqdm import tqdm
import librosa
from torch.utils.data import Dataset, DataLoader
import torchaudio

ROOT_PATH = "F:/DATAS/heartsounds/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/"
postfix = "_New_3주기"
class_names = ["AS", "MR", "MS", "MVP", "N"]
class_label = {"AS": 1, "MR": 2, "MS": 3, "MVP": 4, "N": 0}
data_dirs = [ROOT_PATH + item + postfix + '/' for item in class_names]


# 数据的采样率全是22050，长度分别为：
# AS 57466, MR 46292, MS 65120, MVP 61495, N 46407
# 实际训练的时候，直接取最小值46292吧
def show_data_demo():
    maxlen, minlen = 44100, 44100
    sumlen = 0
    lenlist = []
    for j, item in enumerate(data_dirs):
        print(item)
        for i in range(1, 201):
            ind = ("00" + str(i))[-3:]
            fname = item + f"NEW_{class_names[j]}_{ind}.wav"
            # print(f"NEW_{class_names[j]}_001.wav")
            y, sr = librosa.load(fname)
            if len(y) < minlen:
                minlen = len(y)
            if len(y) > maxlen:
                maxlen = len(y)
            sumlen += len(y)
            lenlist.append(len(y))
            # print(fname, y.shape, sr)
    # min 25482  max 88043  mean 53879.729
    print(minlen, maxlen, sumlen / 1000)


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


def get_wave_label_list(iseval):
    # ind = 0
    train_data_list = []
    train_label_list = []
    valid_data_list = []
    valid_label_list = []
    w2m = Wave2Mel(sr=22050)  # (128,91)
    for j, item in tqdm(enumerate(data_dirs), desc="Reading"):
        label = class_label[class_names[j]]
        for i, rname in enumerate(os.listdir(item)):
            # fname = item + rname
            # data_paths.append(item + rname)
            if i < 180:
                if iseval:
                    continue
                train_data_list.append(w2m(read_wav_mel(item + rname)))
                train_label_list.append(label)
            else:
                valid_data_list.append(w2m(read_wav_mel(item + rname)))
                valid_label_list.append(label)
            # print(ind, fname, label)
            # ind += 1
            # if ind > 4:
            #     return
    # print(ind)
    return train_data_list, train_label_list, valid_data_list, valid_label_list


class HeartDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, ind):
        return self.datas[ind], self.labels[ind]

    def __len__(self):
        return len(self.labels)


def get_loaders(eval=False):
    tdatas, tlabels, vdatas, vlabels = get_wave_label_list(iseval=eval)
    print(vdatas[0].shape)
    if eval:
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)
        return DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)
    else:
        train_dataset = HeartDataset(datas=tdatas,
                                     labels=tlabels)
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)
        return (DataLoader(dataset=train_dataset, batch_size=32, shuffle=True),
                DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False))


if __name__ == '__main__':
    # get_wave_label_list()
    show_data_demo()
    # get_loaders()
