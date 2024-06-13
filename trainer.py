#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/6 14:44
# @Author: ZhaoKe
# @File : HeartSound-Diagnosis.py
# @Software: PyCharm
import os.path
import sys
import time
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(r"C:/Program Files (zk)/PythonFiles/AClassification/Heart-Sound-Diagnosis/")
from models.classifiers import LSTM_Attn_Classifier
from readers import get_loaders


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    # x = torch.randn(16, 1, 64, 128)  # (bs, length, dim)
    # cl_model = LSTM_Classifier(inp_size=87, hidden_size=128, n_classes=5).to(device)
    cl_model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=5,
                                    return_attn_weights=True, attn_type="dot").to(device)
    train_loader, test_loader = get_loaders()
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
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    datetimestr = time.strftime("%Y%m%d%H%M", time.localtime())
    setting_content = "LSTM+DotAttn, adam LambdaLR 0 ~ 5e-4 ~ 5e-5."
    run_save_dir = "./ckpt/data1k/" + datetimestr + f'_/'

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
                # X_mel = X_mel.unsqueeze(1)
                X_mel = X_mel.transpose(1, 2)
            y_mel = y_mel.to(device)
            print(X_mel.shape)
            return
            pred, _ = cl_model(X_mel)
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
                    # X_mel = X_mel.unsqueeze(1)
                    X_mel = X_mel.transpose(1, 2)
                y_mel = y_mel.to(device)
                # print(X_mel.shape)
                pred, _ = cl_model(X_mel)
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
            # if X_mel.ndim == 3:
                # X_mel = X_mel.unsqueeze(1)
                # X_mel = X_mel.transpose(1, 2)
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
    train()
    # heatmap_eval()
