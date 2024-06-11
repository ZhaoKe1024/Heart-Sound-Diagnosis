#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/6 14:44
# @Author: ZhaoKe
# @File : HeartSound-Diagnosis.py
# @Software: PyCharm
import os.path
import sys
import time
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
from models.mobilenetv2 import MobileNetV2
from readers import get_loaders


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    train_loader, test_loader = get_loaders()
    class_loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(cl_model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-5)
    datetimestr = time.strftime("%Y%m%d%H%M", time.localtime())
    run_save_dir = "./ckpt/" + datetimestr + f'_/'

    old = 0
    STD_acc = []
    STD_loss = []
    loss_line = []

    for epoch_id in tqdm(range(60), desc="Train"):
        cl_model.train()
        loss_list = []
        for idx, (X_mel, y_mel) in enumerate(train_loader):
            # print(X_mel.shape, y_mel.shape)
            # return
            optimizer.zero_grad()
            X_mel = X_mel.transpose(1, 2).unsqueeze(1).to(device)
            y_mel = y_mel.to(device)
            # print(X_mel.shape)
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
                X_mel = X_mel.transpose(1, 2).unsqueeze(1).to(device)
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
                if acc_per > 0.75:
                    print(f"Epoch[{epoch_id}]: {acc_per}")
                    if not os.path.exists(run_save_dir):
                        os.makedirs(run_save_dir, exist_ok=True)
                    torch.save(cl_model.state_dict(), run_save_dir+f"cls_model_{epoch_id}.pt")
                    torch.save(optimizer.state_dict(), run_save_dir+f"optimizer_{epoch_id}.pt")
        scheduler.step()
    plt.figure(0)
    plt.plot(range(len(loss_line)), loss_line, c="red", label="train_loss")
    plt.plot(range(len(STD_loss)), STD_loss, c="blue", label="valid_loss")
    plt.plot(range(len(STD_acc)), STD_acc, c="green", label="valid_accuracy")
    plt.xlabel("iteration")
    plt.ylabel("metrics")
    plt.legend()
    plt.savefig(run_save_dir+"train_result.png", format="png", dpi=300)
    plt.show()


def heatmap_eval():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    cl_model.load_state_dict(torch.load(f"./ckpt/202406061626/cls_model_9.pt"))
    test_loader = get_loaders(eval=True)
    ypred_eval = None
    ytrue_eval = None
    acc_list = []
    cl_model.eval()
    with torch.no_grad():
        for jdx, (x_img, y_label) in enumerate(test_loader):
            X_mel = x_img.transpose(1, 2).unsqueeze(1).to(device)
            y_mel = y_label.to(device)
            # print(X_mel.shape)
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
    savepath = "./ckpt/202406061626/result_hm.png"
    max_arg = list(ypred_eval)
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    # conf_mat = conf_mat / conf_mat.sum(axis=1)
    df_cm = pd.DataFrame(conf_mat, index=["N", "AS", "MR", "MS", "MVP"], columns=["N", "AS", "MR", "MS", "MVP"])
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    train()
    # heatmap_eval()
