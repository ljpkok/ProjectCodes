#coding:utf8
import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

## 数据类定义
class VideoDataset(Dataset):
    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        folder = Path(directory)/mode  #获得指定文件夹
        self.clip_len = clip_len

        self.short_side = [128, 160] ##短边缩放区间
        self.crop_size = 112 ##裁剪大小
        self.frame_sample_rate = frame_sample_rate ##帧采样频率，默认为1
        self.mode = mode

        ## 每一个fnames，存储视频文件名
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        ## string与index的映射
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        ## 写入标签
        label_file = str(len(os.listdir(folder)))+'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def __getitem__(self, index):
        ## 读取视频获得图像帧，进行预处理
        buffer = self.loadvideo(self.fnames[index])

        ## 保证读取的视频是有效的
        while buffer.shape[0] < self.clip_len+2 :
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo(self.fnames[index])

        ## 训练时随机翻转
        if self.mode == 'train' or self.mode == 'training':
            buffer = self.randomflip(buffer)

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return buffer, self.label_array[index]

    def to_tensor(self, buffer):
        ## 从[D, H, W, C] 格式转换成 [C, D, H, W] 格式
        return buffer.transpose((3, 0, 1, 2))

    ## 读取视频
    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)##self.frame_sample_rate=1，remainder=0
        ## 读取视频
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ## 根据短边进行随机大小缩放
        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        ## 创建视频buffer，包含frame_count_sample帧
        start_idx = 0
        end_idx = frame_count-1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count>300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        ## 读取图片，输入numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count>end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer
    
    def crop(self, buffer, clip_len, crop_size):
        ## 时间裁剪系数
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        ## 空间裁剪系数
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer                

    def normalize(self, buffer):
        ## 标准化
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        ## 以0.5的概率随机翻转
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':

    datapath = '/hy-tmp/data/UCF-101'
    train_dataloader = \
        DataLoader( VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)
