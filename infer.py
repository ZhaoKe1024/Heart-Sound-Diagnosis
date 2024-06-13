#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/13 10:33
# @Author: ZhaoKe
# @File : infer.py
# @Software: PyCharm
import sys
import torch
sys.path.append(r"C:/Program Files (zk)/PythonFiles/AClassification/Heart-Sound-Diagnosis/")
from models.classifiers import LSTM_Attn_Classifier
from datautils.audioprocess import Wave2Mel, read_wav_mel


def pretrain_setting():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    cl_model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=5,
                                    return_attn_weights=True, attn_type="dot").to(device)
    cl_model.load_state_dict(torch.load(f"./ckpt/data1k/202406120951_LSTMDotAtten/cls_model_60.pt"))
    cl_model.eval()
    w2m = Wave2Mel(sr=22050)
    return w2m, cl_model, device


def test_one():
    dataset_dict = {}
    ROOT_PATH = "F:/DATAS/heartsounds/Classification-of-Heart-Sound-Signal-Using-Multiple-Features--master/"
    dataset_dict["ROOT_PATH"] = ROOT_PATH
    postfix = "_New_3주기"
    class_names = ["N", "AS", "MR", "MS", "MVP"]
    l2m = {}
    m2l = {}
    for i in range(len(class_names)):
        l2m[i] = class_names[i]
        m2l[class_names[i]] = i
    data_dir = [ROOT_PATH + item + postfix + '/' for item in class_names]
    wav_paths = [data_dir[i] + f"New_{class_names[i]}_001.wav" for i in range(5)]

    w2m, cl_model, device = pretrain_setting()
    print(cl_model)
    for input_path in wav_paths:
        X_mel = w2m(read_wav_mel(input_path))
        with torch.no_grad():
            X_mel = X_mel.unsqueeze(0).to(device)  # (87, 128)

            pred, _ = cl_model(X_mel)
            print(f"The prediction of {input_path.split('/')[-1]} is {l2m[pred.argmax(-1).item()]}")
            # print(pred.argmax(-1), pred)

def test_command(input_path):
    class_names = ["N", "AS", "MR", "MS", "MVP"]
    l2m = {}
    m2l = {}
    for i in range(len(class_names)):
        l2m[i] = class_names[i]
        m2l[class_names[i]] = i

    w2m, cl_model, device = pretrain_setting()
    print(cl_model)

    X_mel = w2m(read_wav_mel(input_path))
    with torch.no_grad():
        X_mel = X_mel.unsqueeze(0).to(device)  # (87, 128)

        pred, _ = cl_model(X_mel)
        print(f"The prediction of {input_path.split('/')[-1]} is {l2m[pred.argmax(-1).item()]}")


if __name__ == '__main__':
    # test_one()
    test_command(sys.argv[1])
