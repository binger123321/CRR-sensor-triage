import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.ippa import AAP, AMP
from network.sincnet import SincConv1d


class SincNetModel(nn.Module):
    def __init__(self, use_num):
        super(SincNetModel, self).__init__()

        self.sincnet = torch.nn.Sequential(
            SincConv1d(40, 251, 10000),

            nn.MaxPool1d(kernel_size=3),
            nn.LayerNorm(3250),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.feature_extraction1 = torch.nn.Sequential(
            nn.Conv1d(40, 64, 5, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=3),
            nn.LayerNorm(1082),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(64, 64, 5, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=3),
            nn.LayerNorm(360),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.aap = AAP(in_channel=64, channel=64, reduction=16)
        self.amp = AMP(in_channel=64, channel=64, reduction=16)

        self.fc = nn.Sequential(
            nn.Conv1d(256, 256, 1, stride=1, padding=0),
            nn.MaxPool1d(kernel_size=3),
            nn.LayerNorm(120),
            nn.LeakyReLU(negative_slope=0.2),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifer = torch.nn.Sequential(
            nn.Linear(256 * 1, 3),
        )

    def forward(self, x):
        features = []
        a = []
        for i in range(x.size(1)):
            x_i = x[:, i, :]
            x_i = self.sincnet(x_i)

            x_i = self.feature_extraction1(x_i)

            # ippa 注意力模块
            x_i = self.aap(x_i)
            x_i = self.amp(x_i)

            features.append(x_i)

        output = torch.cat(features, dim=1)  # 8 256,360
        output = self.fc(output)
        output = output.view(output.size(0), -1)
        output = self.classifer(output)

        output = F.softmax(output, dim=-1)
        return output


if __name__ == '__main__':
    start = time.time()
    x = torch.randn([8, 4, 10000]).cuda()
    model = SincNetModel(use_num=4).cuda()
    #
    # # for name, param in model.named_parameters():
    # #     print(f"Layer: {name} | Parameters:{param.numel()}")
    #
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    label = torch.zeros(8).long().cuda()
    outputs = model(x)
    loss = loss_fn(outputs, label)
    #
    loss.backward()
    end = time.time()
    print(f'used time:{end - start}s')
    # print("----")
    # summary(model, input_size=(4, 10000))

    # dummy_input = torch.randn(1, 4, 10000).cuda()
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
