#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/15 12:57
# @Author: ZhaoKe
# @File : PhysioNet2016Dataset.py
# @Software: PyCharm
import os
import pandas as pd
from tqdm import tqdm
from datautils.audioprocess import Wave2Mel, read_wav_mel
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "F:/DATAS/heartsounds/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/"

train_dir = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
valid_dir = ["validation"]
attris = ["2nd left intercostal space",
          "2nd right intercostal space",
          "Apex when sit",
          "Apex when squat",
          "Carotid",
          "Left of apex",
          "Left of parasternum",
          "Parasternum when sit",
          "Parasternum when squat",
          "Superior of apex",
          "Superior of Parasternum"]
attri2label = dict()
for ind, att in enumerate(attris):
    attri2label[att] = ind


def create_attri_file_list():
    output_list = open(f"../datasets/physionet_trainattrilist.txt", 'w')
    df = pd.read_csv(DATA_ROOT + "annotations/Online Appendix_training set.csv", header=0, index_col=None,
                     usecols=[0, 1, 4, 17])
    print(df.head())
    for row in df.itertuples():
        if str(row[4]) != "nan":
            output_list.write('{}.wav,{},{}\n'.format(row[2]+'/'+row[1], row[3], attri2label[row[4].strip()]))
            print(row[1], row[2], row[3], row[4])
    output_list.close()


def create_file_list(mode="train"):
    output_list = open(f"../datasets/physionet_{mode[:5]}list.txt", 'w')
    if mode == "validation":
        read_dir = valid_dir
    else:
        read_dir = train_dir
    for dir in read_dir:
        cur_dir = DATA_ROOT + dir + '/'
        df = pd.read_csv(cur_dir + "REFERENCE.csv", header=None, index_col=None)
        for item in df.itertuples():
            # print(item[0], item[1], item[2])
            # Normal (-1)	Uncertain (0)	Abnormal (1)
            output_list.write('{}.wav,{}\n'.format(dir + '/' + item[1], item[2]))
    output_list.close()


def get_datasets(mode="train"):
    read_file = f"./datasets/physionet_{mode}list.txt"

    data_list = []
    label_list = []
    w2m = Wave2Mel(sr=22050)  # (128,91)
    class_names = ["Abnormal", "Normal"]
    class_label = {"Abnormal": 1, "Normal": 0}
    df = open(read_file, 'r')
    lines = df.readlines()
    for idx, line in tqdm(enumerate(lines), desc=f"Reading_{mode}"):
        parts = line.split(',')
        rname = parts[0]
        label = int(parts[1])
        data_list.append(w2m(read_wav_mel(DATA_ROOT + rname)))
        label_list.append(label)

    df.close()
    return data_list, label_list


class HeartDataset(Dataset):
    def __init__(self, datas, labels, transform=None):
        self.datas = datas
        self.labels = labels
        self.trans = transform

    def __getitem__(self, ind):
        if self.trans is not None:
            return self.trans(self.datas[ind].data.cpu().numpy()), self.labels[ind]
        return self.datas[ind], self.labels[ind]

    def __len__(self):
        return len(self.labels)


def get_loaders(eval=False, bs=32):
    vdatas, vlabels = get_datasets(mode="valid")
    print(len(vdatas), len(vlabels))
    # train_transforms = transforms.Compose([MyRightShift(input_size=(128, 87),
    #                                                     width_shift_range=7,
    #                                                     shift_probability=0.9),
    #                                        # MyAddGaussNoise(input_size=(128, 87),
    #                                        #                 add_noise_probability=0.55),
    #                                        MyReshape(output_size=(1, 128, 87))])
    # test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 87))])
    if eval:
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)  # , transform=test_transforms)
        return DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False)
    else:

        tdatas, tlabels = get_datasets(mode="train")
        print(len(tdatas), len(tlabels))
        train_dataset = HeartDataset(datas=tdatas,
                                     labels=tlabels)  # , transform=train_transforms)
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)  # , transform=test_transforms)
        return (DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True),
                DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False))


if __name__ == '__main__':
    create_attri_file_list()
    # get_loaders()
    # create_file_list("validation")
