#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/7/15 14:32
# @Author: ZhaoKe
# @File : classifiers.py
# @Software: PyCharm
import sys
import os
import time
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim

from datautils.audioprocess import Wave2Mel, read_wav_mel

sys.path.append('../')
sys.path.append('C:/Program Files (zk)/PythonFiles/AClassification/Heart-Sound-Diagnosis/')
from models.classifiers import LSTM_Attn_Classifier
from models.mobilefacenet import MobileFaceNet
from datautils.PhysioNet2016Dataset import get_loaders


class HeartAttriDataset(Dataset):
    def __init__(self, datas, labels, attris, transform=None):
        self.datas = datas
        self.labels = labels
        self.attris = attris
        self.trans = transform

    def __getitem__(self, ind):
        if self.trans is not None:
            return self.trans(self.datas[ind].data.cpu().numpy()), self.labels[ind]
        return self.datas[ind], self.labels[ind], self.attris[ind]

    def __len__(self):
        return len(self.labels)


DATA_ROOT = "F:/DATAS/heartsounds/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/"


def get_attri_loader(bs=32):
    read_file = f"./datasets/physionet_trainattrilist.txt"
    data_list = []
    label_list = []
    attri_list = []
    w2m = Wave2Mel(sr=22050)  # (128,91)
    df = open(read_file, 'r')
    lines = df.readlines()
    for idx, line in tqdm(enumerate(lines), desc=f"ReadingAttri"):
        parts = line.split(',')
        rname = parts[0]
        label = int(parts[1])
        data_list.append(w2m(read_wav_mel(DATA_ROOT + rname)))
        label_list.append(label)
        attri_list.append(int(parts[2]))
    df.close()
    print(len(data_list), len(label_list), len(attri_list))
    attri_dataset = HeartAttriDataset(datas=data_list,
                                      labels=label_list, attris=attri_list)  # , transform=test_transforms)
    return DataLoader(dataset=attri_dataset, batch_size=bs, shuffle=False)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
# x = torch.randn(16, 1, 64, 128)  # (bs, length, dim)
# cl_model = LSTM_Classifier(inp_size=87, hidden_size=128, n_classes=5).to(device)
cl_model = MobileFaceNet(inp_c=1, input_dim=87, latent_size=(6, 8), num_class=2, inp=1).to(device)


# cl_model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=2,
#                                 return_attn_weights=True, attn_type="dot").to(device)


def train():
    train_loader, test_loader = get_loaders()
    attr_loader = get_attri_loader(bs=32)
    class_loss = nn.CrossEntropyLoss().to(device)
    iter_max = 1000
    warm_up_iter, T_max, lr_max, lr_min = 30, iter_max // 3, 5e-4, 5e-5
    # reference: https://blog.csdn.net/qq_36560894/article/details/114004799
    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: 0.005 * cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
                1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    optimizer = optim.Adam(cl_model.parameters(), lr=5e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)

    # 创建一个Embedding层
    vocab_size, embedding_dim = 11, 3
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device)
    embedding_optim = optim.Adam(cl_model.parameters(), lr=5e-3)
    bed_dis = nn.MSELoss().to(device)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    datetimestr = time.strftime("%Y%m%d%H%M", time.localtime())
    setting_content = "MobileFaceNet(inp_c=1, input_dim=87, latent_size=(6, 8), num_class=2, inp=1),\n"
    setting_content += "adam LambdaLR 0 ~ 5e-4 ~ 5e-5,"
    run_save_dir = "./ckpt/physionet/" + datetimestr + f'_/'

    old = 0
    STD_acc = []
    STD_loss = []
    loss_line = []
    lr_list = []
    for epoch_id in tqdm(range(iter_max), desc="Train"):
        cl_model.train()
        loss_list = []
        attri_loss_list = []
        lr_list.append(optimizer.param_groups[0]['lr'])
        for idx, (X_mel, y_mel) in enumerate(train_loader):
            # print(X_mel.shape, y_mel.shape)
            # return
            optimizer.zero_grad()
            X_mel = X_mel.to(device)
            if epoch_id == 0 and idx == 0:
                print(X_mel.shape)
            if X_mel.ndim == 3:
                X_mel = X_mel.transpose(1, 2)
                X_mel = X_mel.unsqueeze(1)
            y_mel = y_mel.to(device)
            # print(y_mel)
            if epoch_id == 0 and idx == 0:
                print(X_mel.shape)
            pred, _ = cl_model(x=X_mel, label=y_mel)
            if epoch_id == 0 and idx == 0:
                # torch.Size([32, 1, 87, 128]) torch.Size([32]) torch.Size([32, 5])
                print(X_mel.shape, y_mel.shape, pred.shape)
            loss_v = class_loss(pred, y_mel)
            loss_v.backward()
            loss_list.append(loss_v.item())
            optimizer.step()
        loss_line.append(np.array(loss_list).mean())

        attri_loss_list_tmp = []
        for idx, (X_mel, y_mel, attr) in enumerate(attr_loader):
            # print(X_mel.shape, y_mel.shape)
            # return
            embedding_optim.zero_grad()
            X_mel = X_mel.to(device)
            if X_mel.ndim == 3:
                X_mel = X_mel.transpose(1, 2)
                X_mel = X_mel.unsqueeze(1)
            y_mel = y_mel.to(device)
            pred, feat = cl_model(x=X_mel, label=y_mel)
            embed = embedding_layer(y_mel.to(torch.long))
            embed = torch.concat((feat[:, :-3], embed), dim=-1)
            if epoch_id == 0 and idx == 0:
                print(embed.shape)
            loss_a = class_loss(pred, y_mel) + bed_dis(feat, embed)
            loss_a.backward()
            attri_loss_list_tmp.append(loss_a.item())
            optimizer.step()
        attri_loss_list.append(np.array(attri_loss_list_tmp).mean())

        cl_model.eval()
        with torch.no_grad():
            acc_list = []
            loss_list = []
            for idx, (X_mel, y_mel) in enumerate(test_loader):
                X_mel = X_mel.to(device)
                if X_mel.ndim == 3:
                    X_mel = X_mel.transpose(1, 2)
                    X_mel = X_mel.unsqueeze(1)
                y_mel = y_mel.to(device)
                # print(X_mel.shape)
                pred, _ = cl_model(x=X_mel, label=y_mel)
                loss_eval = class_loss(pred, y_mel)
                # print(y_mel.argmax(-1))
                # print(pred.argmax(-1))
                acc_batch = metrics.accuracy_score(y_mel.data.cpu().numpy(),
                                                   pred.argmax(-1).data.cpu().numpy())
                acc_list.append(acc_batch)
                loss_list.append(loss_eval.item())
            acc_per = np.array(acc_list).mean()
            # print("new acc:", acc_per)
            STD_acc.append(acc_per)
            STD_loss.append(np.array(loss_list).mean())
            if acc_per > old:
                old = acc_per
                print("new acc:", acc_per)
                if acc_per > 0.65:
                    print(f"Epoch[{epoch_id}]: {acc_per}")
                    if not os.path.exists(run_save_dir):
                        os.makedirs(run_save_dir, exist_ok=True)
                        with open(run_save_dir + f"setting.txt", 'w') as fin:
                            # fin.write("MobileNetV2, adam cosine anneal 5e-4 ~ 5e-5, data augmentation, feature map max reduction.")
                            fin.write(setting_content)
                    torch.save(cl_model.state_dict(), run_save_dir + f"cls_model_{epoch_id}.pt")
                    torch.save(optimizer.state_dict(), run_save_dir + f"optimizer_{epoch_id}.pt")

                    plt.figure(0)
                    plt.subplot(1, 2, 1)
                    plt.plot(range(len(loss_line)), loss_line, c="red", label="train_loss")
                    plt.plot(range(len(STD_loss)), STD_loss, c="blue", label="valid_loss")
                    plt.plot(range(len(STD_acc)), STD_acc, c="green", label="valid_accuracy")
                    plt.xlabel("iteration")
                    plt.ylabel("metrics")
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(lr_list)
                    if not os.path.exists(run_save_dir):
                        os.makedirs(run_save_dir, exist_ok=True)
                    plt.savefig(run_save_dir + f"train_result_{epoch_id}.png", format="png", dpi=300)
                    plt.close()
        scheduler.step()

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_line)), loss_line, c="red", label="train_loss")
    plt.plot(range(len(STD_loss)), STD_loss, c="blue", label="valid_loss")
    plt.plot(range(len(STD_acc)), STD_acc, c="green", label="valid_accuracy")
    plt.xlabel("iteration")
    plt.ylabel("metrics")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(lr_list)
    if not os.path.exists(run_save_dir):
        os.makedirs(run_save_dir, exist_ok=True)
        plt.savefig(run_save_dir + "train_result.png", format="png", dpi=300)
    # plt.close()
    plt.show()


def heatmap_eval():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    cl_model.load_state_dict(torch.load(f"./ckpt/physionet/202407160841_/cls_model_780.pt"))
    test_loader = get_loaders(eval=True)
    ypred_eval = None
    ytrue_eval = None
    acc_list = []
    cl_model.eval()
    with torch.no_grad():
        for jdx, (x_img, y_mel) in enumerate(test_loader):
            X_mel = x_img.to(device)
            if X_mel.ndim == 3:
                X_mel = X_mel.transpose(1, 2)
                X_mel = X_mel.unsqueeze(1)
            y_mel = y_mel.to(device)
            print(X_mel.shape)
            pred, _ = cl_model(x=X_mel, label=y_mel)
            # print(y_label.shape, pred.shape)
            if jdx == 0:
                ytrue_eval = y_mel
                ypred_eval = pred
            else:
                ytrue_eval = torch.concat((ytrue_eval, y_mel), dim=0)
                ypred_eval = torch.concat((ypred_eval, pred), dim=0)
            acc_batch = metrics.accuracy_score(y_mel.data.cpu().numpy(),
                                               pred.argmax(-1).data.cpu().numpy())
            acc_list.append(acc_batch)
            # print(ytrue_eval.shape, ypred_eval.shape)
    print("accuracy:", np.array(acc_list).mean())
    ytrue_eval = ytrue_eval.data.cpu().numpy()
    ypred_eval = ypred_eval.argmax(-1).data.cpu().numpy()
    print(ytrue_eval.shape, ypred_eval.shape)

    # def get_heat_map(pred_matrix, label_vec, savepath):
    savepath = "./ckpt/data1k/202406120951_LSTMDotAtten/result_hm.png"
    max_arg = list(ypred_eval)
    print(metrics.precision_score(max_arg, ytrue_eval))
    print(metrics.recall_score(max_arg, ytrue_eval))
    print(metrics.f1_score(max_arg, ytrue_eval))
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    print(conf_mat)
    conf_mat = conf_mat / conf_mat.sum(axis=1)
    ab2full = ["Normal \nHeart Sound", "Abnormal\nHeart Sound"]
    # ab2full = ["A Normal \nHeart Sound", "Aortic\nStenosis", "Mitral\nRegurgitation", "Mitral\nStrnosis", "Murmur\nin Systole"]
    df_cm = pd.DataFrame(conf_mat, index=ab2full, columns=ab2full)
    # heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    # heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')  # , cbar_kws={'format': '%.2f%'})
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    # plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    # train()
    heatmap_eval()
