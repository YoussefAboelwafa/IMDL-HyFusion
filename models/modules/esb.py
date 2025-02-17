from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from resnet import ResNet50



def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res



class ESB(nn.Module):

    def __init__(self, num_classes, sobel, n_input=3, pretrained = True, **kwargs):
        super(ESB, self).__init__()
        self.sobel = sobel
        self.num_classes = num_classes
        self.backbone = ResNet50(pretrained=pretrained, n_input=n_input)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.erb_db_1 = ERB(256, self.num_classes)
        self.erb_db_2 = ERB(512, self.num_classes)
        self.erb_db_3 = ERB(1024, self.num_classes)
        self.erb_db_4 = ERB(2048, self.num_classes)

        self.erb_trans_1 = ERB(self.num_classes, self.num_classes)
        self.erb_trans_2 = ERB(self.num_classes, self.num_classes)
        self.erb_trans_3 = ERB(self.num_classes, self.num_classes)
        if self.sobel:
            print("----------use sobel-------------")
            self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(512, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(1024, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(2048, 1)

    def forward(self, x):
        input_ = x.clone()
        feature_map, _ = self.backbone(input_)
        c1, c2, c3, c4 = feature_map

        if self.sobel:
            res1 = self.erb_db_1(run_sobel(self.sobel_x1, self.sobel_y1, c1))
            res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(run_sobel(self.sobel_x2, self.sobel_y2, c2))))
            res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(run_sobel(self.sobel_x3, self.sobel_y3, c3))))
            res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(run_sobel(self.sobel_x4, self.sobel_y4, c4))), relu=False)

        else:
            res1 = self.erb_db_1(c1)
            res1 = self.erb_trans_1(res1 + self.upsample(self.erb_db_2(c2)))
            res1 = self.erb_trans_2(res1 + self.upsample_4(self.erb_db_3(c3)))
            res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)
        
        return res1 , c4
    