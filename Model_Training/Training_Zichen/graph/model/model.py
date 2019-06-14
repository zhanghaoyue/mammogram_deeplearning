import numpy as np
import torch
import torch.nn.functional as F

from .base_model import *


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        #         print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        #         print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        #         print(x3.shape)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        #         print(x4.shape)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        #         print(x5.shape)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        #         print(d5.shape)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #         print(d4.shape)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #         print(d3.shape)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        #         print(d2.shape)

        d1 = self.Conv_1x1(d2)
        #         print(d1.shape)

        attention_map = F.sigmoid(d1[:, 0, :, :])

        return attention_map


class AttU_Net_Classification(nn.Module):
    def __init__(self, img_ch=1):
        super(AttU_Net_Classification, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.adaptive_avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classification = nn.Sequential(nn.Conv2d(1024, 16, kernel_size=3, stride=2, padding=1, bias=True),
                                            nn.MaxPool2d(kernel_size=4, stride=4),
                                            )
        self.fc = nn.Sequential(nn.Dropout(),
                                nn.Linear(16 * 4 * 4, 1),
                                nn.Sigmoid())

    def forward(self, x, mass_region_attention):
        x = x * (0.1 + mass_region_attention)
        # encoding path
        x1 = self.Conv1(x)
        # print(x1.shape)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print(x2.shape)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print(x3.shape)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print(x4.shape)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # print(x5.shape)

        x6 = self.classification(x5)

        x7 = x6.view(x6.shape[0], -1)

        prob = self.fc(x7)

        return prob


if __name__ == '__main__':
    model = AttU_Net_Classification(img_ch=1)
    img = torch.FloatTensor(np.ones([2, 1, 512, 512]))
    attention_map = torch.FloatTensor(np.ones([2, 1, 512, 512]))
    prob = model(img, attention_map)
    print(prob)
