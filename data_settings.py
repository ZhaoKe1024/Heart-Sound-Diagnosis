#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/11 19:50
# @Author: ZhaoKe
# @File : dataset_dircor.py
# @Software: PyCharm
import librosa

def dataset_circor():
    dataset_dict = {}
    ROOT_PATH = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/"
    metadata = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data.csv"
    dataset_dict["ROOT_PATH"] = ROOT_PATH
    dataset_dict["metadata"] = metadata
    postfix = "_New_3주기"
    class_names = ["AV", "MV", "PV", "TV", "Phc"]
    dataset_dict["class_names"] = class_names
    dataset_dict["class_label"] = {"AV": 1, "MV": 2, "PV": 3, "TV": 4, "Phc": 5}
    dataset_dict["data_dirs"] = [ROOT_PATH + item + postfix + '/' for item in class_names]
    return dataset_dict


def dataset_1k():
    dataset_dict = {}
    ROOT_PATH = "F:/DATAS/heartsounds/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/"
    dataset_dict["ROOT_PATH"] = ROOT_PATH
    postfix = "_New_3주기"
    class_names = ["AS", "MR", "MS", "MVP", "N"]
    dataset_dict["class_names"] = class_names
    dataset_dict["class_label"] = {"AS": 1, "MR": 2, "MS": 3, "MVP": 4, "N": 0}
    dataset_dict["data_dirs"] = [ROOT_PATH + item + postfix + '/' for item in class_names]
    return dataset_dict


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
