from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import os
import skvideo
import skvideo.io
from torch.utils.data.sampler import Sampler
from random import sample
import torchvision.transforms as transforms

## 创建transform
def set_transforms(mode):
    if mode == 'train':
        transform = transforms.Compose(
            [transforms.Resize(256),  # this is set only because we are using Imagenet pre-train model.
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
             ])
    elif mode == 'test' or mode == 'val':
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))])
    return transform

## 数据读取类
class UCF101Dataset(Dataset):
    def __init__(self, data_path, data, mode, dataset='UCF101'):
        super(UCF101Dataset, self).__init__()
        self.dataset = dataset
        if self.dataset == 'UCF101':
            self.labels = data[1]
        self.data_path = data_path
        self.images = data[0]
        self.transform = set_transforms(mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.dataset == 'UCF101':
            sampled_video_name = self.images[idx].split('/')[1] +'.avi'
        elif self.dataset == 'youtube':
            sampled_video_name = self.images[idx]
        else:
           print('You have enter a wrong dataset type in the dataset function. please fix it. possabilites are youtube or UCF101(the default)')

        # ====== 使用skvideo读取视频帧并进行预处理 =======
        video_frames = skvideo.io.vread(os.path.join(self.data_path, sampled_video_name))
        video_frames_array = []
        for image in video_frames:
            img = Image.fromarray(image.astype('uint8'), 'RGB')
            img = self.transform(img)
            video_frames_array.append(img)
        img_stack = torch.stack(video_frames_array) ##将list拼接为高维张量=5*3*224*224
        if self.dataset == 'UCF101':
            label = torch.from_numpy(np.asarray(int(self.labels[idx]))).long()
            return img_stack, label, idx
        else:
            return img_stack

class UCF101DatasetSampler(Sampler):
    def __init__(self, data, batch_size):
        self.num_samples = len(data)
        self.classes_that_were_sampled = []
        self.data_labels = data.labels
        self.batch_size = batch_size

    def __iter__(self):
        idx_list = []
        for i in range(self.batch_size):
            idx_image_sample = sample(range(self.num_samples), 1)[0]
            label_sample = self.data_labels[idx_image_sample]
            while label_sample in self.classes_that_were_sampled:
                idx_image_sample = sample(range(self.num_samples), 1)[0]
                label_sample = self.data_labels[idx_image_sample]
            self.classes_that_were_sampled += [label_sample]
            idx_list += [idx_image_sample]
        return iter(idx_list)

    def __len__(self):
        return self.num_samples









