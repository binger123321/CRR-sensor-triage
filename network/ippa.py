import torch
from torch import nn


class AAP(nn.Module):
    def __init__(self, in_channel, channel, reduction=16):
        super(AAP, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel * 7, channel * 7 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 7 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy

        y2 = self.avg_pool2(x)
        y2 = y2.view(b, 2 * c)  # 4,64

        y3 = self.avg_pool4(x)  # Broadcasting
        y3 = y3.view(b, 4 * c)  # 4,128

        y = torch.cat((y1, y2, y3), 1)  # 4,224
        # 降维，生成注意力向量
        y = self.fc(y).unsqueeze(2)
        x = x * y.expand_as(x)
        return x


class AMP(nn.Module):
    def __init__(self, in_channel, channel, reduction=16):
        super(AMP, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(channel),
                nn.ReLU(inplace=True)
            )
        self.max_pool1 = nn.AdaptiveMaxPool1d(1)
        self.max_pool2 = nn.AdaptiveMaxPool1d(2)
        self.max_pool4 = nn.AdaptiveMaxPool1d(4)

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(channel * 7, channel * 7 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 7 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _ = x.size()

        y1 = self.max_pool1(x).view(b, c)  # Max pooling

        y2 = self.max_pool2(x)
        y2 = y2.view(b, 2 * c)

        y3 = self.max_pool4(x)
        y3 = y3.view(b, 4 * c)

        y = torch.cat((y1, y2, y3), 1)
        # 降维，生成注意力向量
        y = self.fc(y).unsqueeze(2)
        x = x * y.expand_as(x)
        return x


if __name__ == '__main__':
    x = torch.randn([4, 64, 360])
    b, c, n = x.shape
    aap = AAP(in_channel=64, channel=64, reduction=16)
    amp = AMP(in_channel=64, channel=64, reduction=16)
    out = aap(x)
    out = amp(out)
    print(out.shape)
