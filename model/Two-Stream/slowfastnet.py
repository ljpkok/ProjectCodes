#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['resnet50', 'resnet101','resnet152', 'resnet200']

## 瓶颈模块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        ## 根据是否在时间上进行卷积，有两种配置
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

## 模型定义
class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5 ):
        super(SlowFast, self).__init__()
         
        ## fast分支定义
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        ## fast残差模块，时间卷积核大小为3
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)
        
        ## fast与slow分支进行融合前的变换
        self.lateral_p1 = nn.Conv3d(8, 8*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res2 = nn.Conv3d(32,32*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res3 = nn.Conv3d(64,64*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))
        self.lateral_res4 = nn.Conv3d(128,128*2, kernel_size=(5, 1, 1), stride=(8, 1 ,1), bias=False, padding=(2, 0, 0))

        ## slow分支定义
        self.slow_inplanes = 64+64//8*2
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        ## slow残差模块，前两层时间卷积核大小为1，后两层时间卷积核大小为3
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes+2048, class_num, bias=False)

    def forward(self, input):
        fast, lateral = self.FastPath(input[:, :, ::2, :, :]) ##时间步长2
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral) ##时间步长16
        x = torch.cat([slow, fast], dim=1) ##最后的特征拼接
        x = self.dp(x)
        x = self.fc(x)
        return x

    ## slow分支，需要输入fast分支的特征
    def SlowPath(self, input, lateral):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, lateral[0]],dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, lateral[1]],dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, lateral[2]],dim=1)
        x = self.slow_res4(x)
        x = torch.cat([x, lateral[3]],dim=1)
        x = self.slow_res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    ## fast分支
    def FastPath(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_res2(res2)
        lateral.append(lateral_res2)
        
        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    ## fast残差模块
    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    ## slow残差模块
    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
  
        self.slow_inplanes = planes * block.expansion + planes * block.expansion//8*2
        return nn.Sequential(*layers)

def resnet20(**kwargs):
    """Constructs a ResNet-20 model.
    """
    model = SlowFast(Bottleneck, [1, 2, 2, 1], **kwargs)
    return model

def resnet32(**kwargs):
    """Constructs a ResNet-32 model.
    """
    model = SlowFast(Bottleneck, [2, 3, 3, 2], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == "__main__":
    num_classes = 101
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 64, 224, 224))
    model = resnet50(class_num=num_classes)
    output = model(input_tensor)
    print(output.size())
