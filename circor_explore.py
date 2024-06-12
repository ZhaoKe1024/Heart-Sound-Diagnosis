# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-06-05 21:35
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import soundfile
from datautils.audioprocess import Wave2Mel


def read_txt():
    pass


def read_tsv(filepath):
    mat = []
    with open(filepath, 'r') as fin:
        line = fin.readline()
        while line:
            parts = line.split('\t')
            mat.append([float(parts[0]), float(parts[1]), int(parts[2])])
            line = fin.readline()
    return mat


def read_hea(filepath):
    # mat = []
    with open(filepath, 'r') as fin:
        line = fin.readline()
        while line:
            print(line)
            # parts = line.split('\t')
            # mat.append([float(parts[0]), float(parts[1]), int(parts[2])])
            line = fin.readline()
    # return mat


def read_segments(wav_path):
    y, sr = librosa.load(wav_path)
    print(librosa.get_duration(y, sr))
    pass


# tem_files("2530", "AV+PV+TV+MV"))
# print(get_item_files("29378", "AV+MV"))
# print(get_im_files("46532", "AV"))
def get_item_files(p_id: str, locs: str, onlywav=True):
    exts = ["hea", "tsv", "wav"]
    flist = []
    if '+' in locs:
        parts = locs.split('+')
        for part in parts:
            flist.append(p_id + '_' + part)
    res_list = []

    # for ext in exts:
    #     if len(flist) == 0:
    #         res_list.append(p_id + '_' + locs  + '.' + ext)
    #     else:
    #         for fname in flist:
    #             res_list.append(p_id + '_' + fname + '.' + ext)

    if len(flist) == 0:
        for ext in exts:
            res_list.append(p_id + '_' + locs + '.' + ext)
    else:
        for fname in flist:
            for ext in exts:
                res_list.append(fname + '.' + ext)
    return res_list


def read_data():
    label2num = {"Normal": 0, "Abnormal": 1}
    data_root = "D:/DATAS/Medical/circor/"
    data_dir = "training_data/"
    metafile = "training_data.csv"
    metadata = pd.read_csv(data_root + metafile, header=0, index_col=None, usecols=[0, 1, 20])
    metadata["Outcome"] = metadata["Outcome"].apply(lambda x: label2num[x])
    # print(metadata)
    for j, row in metadata.iterrows():
        # print(row[0], row[1])
        label = row[2]
        cur_files = get_item_files(str(row[0]), row[1])
        print(cur_files[0], label)


def build_wavdatalist_circor():
    label2num = {"Normal": 0, "Abnormal": 1}
    data_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/"
    data_dir = "training_data/"
    metafile = "training_data.csv"
    metadata = pd.read_csv(data_root + metafile, header=0, index_col=None, usecols=[0, 1, 20])
    metadata["Outcome"] = metadata["Outcome"].apply(lambda x: label2num[x])
    # print(metadata)
    flist = []
    label_list = []
    for j, row in metadata.iterrows():
        # print(row[0], row[1])
        label = row[2]
        cur_files = get_item_files(str(row[0]), row[1])
        locs = row[1]
        if '+' in locs:
            parts = locs.split('+')
            for part in parts:
                flist.append(str(row[0]) + '_' + part + ".wav")
                label_list.append(label)
        else:
            flist.append(str(row[0]) + '_' + locs + ".wav")
            label_list.append(label)

        # print(cur_files[0], label)
    # print(flist)
    print(len(label_list), np.array(label_list).shape, np.array(label_list).sum())
    with open("./wavdatalist_circor.txt", 'w') as fin:
        for i in range(len(label_list)):
            fin.write(flist[i] + ',' + str(label_list[i]) + '\n')


def segment_wav(sig, length=198450, overlap=66150, sr=22050):
    st, end = 0, length
    segments = []
    while end < len(sig):
        segments.append(sig[st:end])
        st += length-overlap
        end = st + length
    if len(sig) - st > length-overlap:
        tmp = np.zeros(length)
        st1 = (length - (len(sig)-st))//2
        tmp[st1:st1+len(sig)-st] = sig[st:]
        segments.append(tmp)
    # print(len(segments))
    return segments


def read_wav():
    data_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/"
    new_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data_9s/"
    cnt = 0
    length_list = []
    segs = []
    fin1 = open("./datasets/wavdatalist_circor_9s.txt", 'w')
    with open("./datasets/wavdatalist_circor.txt", 'r') as fout:
        lines = fout.readlines()
        for j, line in tqdm(enumerate(lines), desc="Reading"):
            parts = line.split(',')
            # print(data_root + parts[0])
            if os.path.exists(data_root + parts[0]):
                y, sr = librosa.load(path=data_root + parts[0])
                segments = segment_wav(y)

                for idx, seg in enumerate(segments):
                    fin1.write(parts[0][:-4]+'_' + ("00" + str(idx))[-2:] + ".wav,"+parts[1])
                    soundfile.write(new_root + parts[0][:-4]+'_' + ("00" + str(idx))[-2:] + ".wav", seg, samplerate=22050)
                # print(len(segments), parts[0])
                # segs.extend(segment_wav(y))
                # lengtgh = len(y) / sr
                # length_list.append(lengtgh)
                # cnt += 1
                # print(lengtgh, sr, len(y))
            else:
                y, sr = librosa.load(path=data_root + parts[0][:-4]+'_1.wav')
                segments = segment_wav(y)
                for idx, seg in enumerate(segments):
                    fin1.write(parts[0][:-4]+'_1_' + ("00" + str(idx))[-2:] + ".wav,"+parts[1])
                    soundfile.write(new_root + parts[0][:-4]+'_1_' + ("00" + str(idx))[-2:] + ".wav", seg, samplerate=22050)
                # segs.extend(segment_wav(y))
                # lengtgh = len(y) / sr
                # length_list.append(lengtgh)
                # cnt += 1

                y, sr = librosa.load(path=data_root + parts[0][:-4]+'_2.wav')
                segments = segment_wav(y)
                for idx, seg in enumerate(segments):
                    fin1.write(parts[0][:-4]+'_2_' + ("00" + str(idx))[-2:] + ".wav,"+parts[1])
                    soundfile.write(new_root + parts[0][:-4]+'_2_' + ("00" + str(idx))[-2:] + ".wav", seg, samplerate=22050)
                # segs.extend(segment_wav(y))
                # lengtgh = len(y) / sr
                # length_list.append(lengtgh)
                # cnt += 1

                # print(lengtgh, sr, len(y))
            #     continue
            # if j > 1 and j % 50 == 0:
            #     # break
            #     print(np.array(length_list).mean())
    # print(length_list)
    print(cnt, len(segs))
    lengths = np.array(length_list)
    print(cnt * lengths.mean())
    print(lengths.sum())
    fin1.close()


def read_wav_9s():
    data_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data_9s/"
    w2m = Wave2Mel(sr=22050)
    label_list = []
    pos_cnt, neg_cnt = 0, 0
    with open("./datasets/wavdatalist_circor_9s.txt", 'r') as fout:
        lines = fout.readlines()
        for j, line in tqdm(enumerate(lines), desc="Reading"):
            parts = line.split(',')
            # print(data_root + parts[0])
            if int(parts[1]):
                pos_cnt += 1
            else:
                neg_cnt += 1
            # label_list.append(parts[1])
            # y, sr = librosa.load(path=data_root + parts[0])
            #
            # logmel = w2m(torch.from_numpy(y[:147000]))
            # print(sr, logmel.shape)
            # if j == 10:
            #     break
    print(pos_cnt, neg_cnt)
    # print(np.array(label_list).sum())
    # print(np.array(label_list))


if __name__ == '__main__':
    # read_data()

    data_path = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data_9s/9983_MV_01.wav"
    y, sr = librosa.load(data_path)
    w2m = Wave2Mel(sr=22050)
    mel = w2m(torch.from_numpy(y))
    plt.figure(0)
    plt.imshow(mel)
    plt.show()
    # read_wav_9s()
