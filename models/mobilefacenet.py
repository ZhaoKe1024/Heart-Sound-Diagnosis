# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-21 22:18
import math
import torch.nn as nn
from models.loss import ArcMarginProduct


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, inp_c=2, input_dim=128, latent_size=(14, 4), bottleneck_setting=Mobilefacenet_bottleneck_setting, num_class=6, inp=1):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(inp_c, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        # self.linear7 = ConvBlock(512, 512, (8, 18), 1, 0, dw=True, linear=True)
        self.linear7 = ConvBlock(512, 512, latent_size, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, input_dim*2)
        self.cls = ArcMarginProduct(in_features=input_dim*2, out_features=num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        # print("shape of bottleneck output:", x.shape)
        x = self.conv2(x)
        # print("shape of convblock output:", x.shape)
        x = self.linear7(x)
        # print("shape of convblock output:", x.shape)
        x = self.linear1(x)
        # print("shape of convblock output:", x.shape)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        # print(out.shape, label.shape)
        out = self.cls(out, label)
        return out, feature


if __name__ == '__main__':
    import torch
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cl_model = MobileFaceNet(inp_c=1, input_dim=87, latent_size=(6, 8), num_class=2, inp=1).to(device)
    x = torch.randn(16, 1, 87, 128, device=device)  # (bs, length, dim)
    label = torch.randint(low=0, high=2, size=(16,), device=device)
    # cl_model = LSTM_Classifier(inp_size=87, hidden_size=128, n_classes=5).to(device)
    # cl_model = LSTM_Attn_Classifier(inp_size=87, hidden_size=128, n_classes=2,
    #                                 return_attn_weights=True, attn_type="dot").to(device)
    tmp_pred, tmp_feat = cl_model(x, label)
    class_loss = nn.CrossEntropyLoss().to(device)
    loss_v = class_loss(tmp_pred, label)
    loss_v.backward()
    print(tmp_pred.shape, tmp_feat.shape)
