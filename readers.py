#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/6 14:46
# @Author: ZhaoKe
# @File : readers.py
# @Software: PyCharm
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datautils.audioprocess import Wave2Mel, read_wav_mel
from data_settings import dataset_1k


def get_wave_label_list(iseval):
    # ind = 0
    train_data_list = []
    train_label_list = []
    valid_data_list = []
    valid_label_list = []
    w2m = Wave2Mel(sr=22050)  # (128,91)
    config = dataset_1k()
    class_label, class_names = config["class_label"], config["class_names"]
    for j, item in tqdm(enumerate(config["data_dirs"]), desc="Reading"):
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


def get_loaders(eval=False, bs=16):
    tdatas, tlabels, vdatas, vlabels = get_wave_label_list(iseval=eval)
    print(vdatas[0].shape)

    # train_transforms = transforms.Compose([MyRightShift(input_size=(128, 87),
    #                                                     width_shift_range=7,
    #                                                     shift_probability=0.9),
    #                                        # MyAddGaussNoise(input_size=(128, 87),
    #                                        #                 add_noise_probability=0.55),
    #                                        MyReshape(output_size=(1, 128, 87))])
    # test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 87))])

    if eval:
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)#, transform=test_transforms)
        return DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False)
    else:
        train_dataset = HeartDataset(datas=tdatas,
                                     labels=tlabels)#, transform=train_transforms)
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)#, transform=test_transforms)
        return (DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True),
                DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False))


if __name__ == '__main__':
    # get_wave_label_list()
    # show_data_demo()
    get_loaders()
