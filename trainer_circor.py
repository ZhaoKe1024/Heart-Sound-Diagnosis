#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/12 18:39
# @Author: ZhaoKe
# @File : trainer_circor.py
# @Software: PyCharm
import os.path
import sys
import time
import math
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
sys.path.append(r"C:/Program Files (zk)/PythonFiles/AClassification/Heart-Sound-Diagnosis/")
from models.mobilefacenet import MobileFaceNet
from models.classifiers import LSTM_Attn_Classifier
from datautils.audioprocess import Wave2Mel


class HeartDataset(Dataset):
    def __init__(self, datas, labels, transform=None):
        self.datas = datas
        self.labels = labels
        self.trans = transform

    def __getitem__(self, ind):
        return self.datas[ind], self.labels[ind]

    def __len__(self):
        return len(self.labels)


def get_loaders(eval=False, bs=16):
    data_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data_9s/"
    mel_root = "F:/DATAS/heartsounds/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data_9smel/"
    os.makedirs(mel_root, exist_ok=True)
    sr = 22050
    mel_size = (56, 224)
    w2m = Wave2Mel(sr=sr)
    vdatas, vlabels = [], []
    tdatas, tlabels = [], []
    vpos, vneg = 0, 0
    with open("./datasets/wavdatalist_circor_9s.txt", 'r') as fout:
        lines = fout.readlines()
        for j, line in tqdm(enumerate(lines), desc="Reading"):
            parts = line.split(',')
            lb = int(parts[1])
            if (vneg < 200 and lb == 0) or (vpos < 200 and lb == 1):
                y, sr = librosa.load(path=data_root + parts[0])
                logmel = w2m(torch.from_numpy(y[:147000]))
                logmel = logmel[:mel_size[0], 64:]
                # print(y.shape, logmel.shape)
                # new_im = Image.fromarray(logmel.data.numpy().astype(np.uint8))
                # plt.figure(j)
                # plt.imshow(new_im)
                # plt.xticks([])
                # plt.yticks([])
                # plt.savefig(mel_root+parts[0][:-4]+".png", dpi=300, format="png")
                # plt.close(j)

                vdatas.append(logmel)
                vlabels.append(lb)
                if lb == 0:
                    vneg += 1
                else:
                    vpos += 1
            else:
                y, sr = librosa.load(path=data_root + parts[0])
                logmel = w2m(torch.from_numpy(y[:147000]))
                tdatas.append(logmel[:mel_size[0], 64:])
                tlabels.append(int(parts[1]))
            if j==1000:
                break
    print(len(tdatas), len(tlabels))
    print(len(vdatas), len(vlabels))
    if eval:
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)
        return DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False)
    else:
        train_dataset = HeartDataset(datas=tdatas,
                                     labels=tlabels)
        valid_dataset = HeartDataset(datas=vdatas,
                                     labels=vlabels)
        return (DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True),
                DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False))


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    # x = torch.randn(16, 1, 64, 128)  # (bs, length, dim)
    input_size = (56, 224)
    # cl_model = LSTM_Attn_Classifier(inp_size=input_size[0], hidden_size=128, n_classes=2, return_attn_weights=True, attn_type="dot").to(device)
    cl_model = MobileFaceNet(inp_c=1, input_dim=input_size[0], latent_size=(14, 4), num_class=2, inp=1).to(device)
    train_loader, test_loader = get_loaders(eval=False, bs=32)
    class_loss = nn.CrossEntropyLoss().to(device)
    iter_max = 100
    warm_up_iter, T_max, lr_max, lr_min = 10, iter_max // 2.5, 5e-3, 5e-4
    # reference: https://blog.csdn.net/qq_36560894/article/details/114004799
    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: lr_max * cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
                1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    optimizer = optim.Adam(cl_model.parameters(), lr=lr_max)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    datetimestr = time.strftime("%Y%m%d%H%M", time.localtime())
    setting_content = "LSTM+DotAttn, adam LambdaLR 0 ~ 5e-4 ~ 5e-5."
    run_save_dir = "./ckpt/data_circor/" + datetimestr + f'_/'

    old = 0
    STD_acc = []
    STD_loss = []
    loss_line = []
    lr_list = []

    for epoch_id in tqdm(range(iter_max), desc="Train"):
        cl_model.train()
        loss_list = []
        lr_list.append(optimizer.param_groups[0]['lr'])
        for idx, (X_mel, y_mel) in enumerate(train_loader):
            # print(X_mel.shape, y_mel.shape)
            # return
            optimizer.zero_grad()
            X_mel = X_mel.to(device)
            if X_mel.ndim == 3:
                X_mel = X_mel.transpose(1, 2)
                X_mel = X_mel.unsqueeze(1)
            y_mel = y_mel.to(device)
            if epoch_id < 2 and idx < 2:
                print(X_mel.shape, y_mel.shape)
            # return
            pred, _ = cl_model(x=X_mel, label=y_mel)
            # print(pred.shape)
            # return
            # if idx == 0:
            #     # torch.Size([32, 1, 87, 128]) torch.Size([32]) torch.Size([32, 5])
            #     print(X_mel.shape, y_mel.shape, pred.shape)
            loss_v = class_loss(pred, y_mel)
            loss_v.backward()
            loss_list.append(loss_v.item())
            optimizer.step()
        loss_line.append(np.array(loss_list).mean())
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
                if acc_per > 0.85:
                    print(f"Epoch[{epoch_id}]: {acc_per}")
                    if not os.path.exists(run_save_dir):
                        os.makedirs(run_save_dir, exist_ok=True)
                        with open(run_save_dir + f"setting.txt", 'w') as fin:
                            # fin.write("MobileNetV2, adam cosine anneal 5e-4 ~ 5e-5, data augmentation, feature map max reduction.")
                            fin.write(setting_content)
                    torch.save(cl_model.state_dict(), run_save_dir + f"cls_model_{epoch_id}.pt")
                    torch.save(optimizer.state_dict(), run_save_dir + f"optimizer_{epoch_id}.pt")
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
    plt.show()


def heatmap_eval():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    cl_model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=5,
                                    return_attn_weights=True, attn_type="dot").to(device)
    cl_model.load_state_dict(torch.load(f"./ckpt/data1k/202406120951_LSTMDotAtten/cls_model_60.pt"))
    test_loader = get_loaders(eval=True)
    ypred_eval = None
    ytrue_eval = None
    acc_list = []
    cl_model.eval()
    with torch.no_grad():
        for jdx, (x_img, y_label) in enumerate(test_loader):
            X_mel = x_img.to(device)
            if X_mel.ndim == 3:
                # X_mel = X_mel.unsqueeze(1)
                X_mel = X_mel.transpose(1, 2)
            y_mel = y_label.to(device)
            print(X_mel.shape)
            pred, _ = cl_model(X_mel)
            # print(y_label.shape, pred.shape)
            if jdx == 0:
                ytrue_eval = y_label
                ypred_eval = pred
            else:
                ytrue_eval = torch.concat((ytrue_eval, y_label), dim=0)
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
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    conf_mat = conf_mat / conf_mat.sum(axis=1)
    df_cm = pd.DataFrame(conf_mat, index=["N", "AS", "MR", "MS", "MVP"], columns=["N", "AS", "MR", "MS", "MVP"])
    # heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    # get_loaders()
    train()
    # heatmap_eval()
