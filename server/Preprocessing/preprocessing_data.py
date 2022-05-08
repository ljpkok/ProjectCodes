# coding:utf8
import argparse
from tqdm import tqdm
from utils_action_recognition import save_setting_info
import numpy as np
from tqdm import tnrange, tqdm_notebook  # used when I run in colab/GCloud
import os
import cv2
import math
from PIL import Image
import pickle
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='UCF101 Action Recognition preprocessing data, LRCN architecture')
parser.add_argument('--row_data_dir', default=r'/hy-tmp/data/UCF-101', type=str,
                    help='path to find the UCF101 row data')
parser.add_argument('--ucf_list_dir',
                    default=r'D:\BaiduYunDownload\UCF-101-Video\ucfTrainTestlist',
                    type=str, help='path to find the UCF101 list splitting the data to train and test')
parser.add_argument('--sampling_rate', default=10, type=int, help='how to sample the data')
parser.add_argument('--ucf101_fps', default=25, type=int, help='FPS of the UCF101 dataset')
parser.add_argument('--num_frames_to_extract', default=5, type=int,
                    help='The number of frames what would be extracted from each video')
parser.add_argument('--video_file_name', default='y2mate.com - cute_happy_baby_crawling_BkJ6FJ2jJEQ_360p.mp4', type=str,
                    help='the video file name we would process, if none the script would run on all of the video files in the folder')
parser.add_argument('--dataset', default='UCF101', type=str,
                    help='the dataset name. options = youtube, UCF101')


## 对视频进行预处理，使其长度相等
def main_procesing_data(args, folder_dir, sampled_video_file=None, processing_mode='main'):
    """"
       Create the sampled data video,
       input - video, full length.
       function - 1. 使用opencv读取视频，按照处以args.sampling_rate，降低视频帧率
                  2. 随机设置起始帧，然后接下来Y(args.num_frames_to_extract)个连续帧被提取
                  3. 如果模式为processing_mode == 'main'，提取y个连续帧，存到新视频；如果否，则传给下一个函数
       Output: 输出新的视频
       """
    if args.dataset == 'UCF101':
        for file_name in os.listdir(args.ucf_list_dir):
            # ===== 按照split 1划分方式进行划分 =====
            if '1' in file_name:
                with open(os.path.join(args.ucf_list_dir, file_name)) as f:
                    video_list = f.readlines()
                with tqdm(total=len(video_list)) as pbar:
                    for video_name in video_list:
                        video_name = video_name.split(' ')[0].rstrip('\n')
                        capture_and_sample_video(args.row_data_dir, video_name, args.num_frames_to_extract,
                                                 args.sampling_rate, args.ucf101_fps, folder_dir,
                                                 args.ucf101_fps, processing_mode)
                        pbar.update(1)
            else:
                pass

    elif args.dataset == 'youtube':
        video_original_size_dict = {}
        if args.video_file_name is None and sampled_video_file is None:
            for file_name in os.listdir(args.row_data_dir):
                video_test, video_original_size = capture_and_sample_video(args.row_data_dir, file_name, 'all',
                                                                           args.sampling_rate, 'Not known', folder_dir,
                                                                           args.ucf101_fps, processing_mode)
                video_original_size_dict[file_name.split('.mp4')[0]] = video_original_size
        else:
            file_name = args.video_file_name if sampled_video_file is None else sampled_video_file[0]
            video_test, video_original_size = capture_and_sample_video(args.row_data_dir, file_name, 'all',
                                                                       args.sampling_rate, 'Not known', folder_dir,
                                                                       args.ucf101_fps, processing_mode)
            video_original_size_dict[file_name.split('.mp4')[0]] = video_original_size

        with open(os.path.join(folder_dir, 'video_original_size_dict.pkl'), 'wb') as f:
            pickle.dump(video_original_size_dict, f, pickle.HIGHEST_PROTOCOL)
        if processing_mode == 'live':
            return video_test, video_original_size


def main_processing_data2(row_data_dir, video_file_name, sampling_rate, folder_dir):
    video_original_size_dict = {}
    video_test, video_original_size = capture_and_sample_video(row_data_dir, video_file_name, 'all',
                                                               sampling_rate, 'Not known', folder_dir,
                                                               25, "live")
    video_original_size_dict[video_file_name.split('.mp4')[0]] = video_original_size
    with open(os.path.join(folder_dir, 'video_original_size_dict.pkl'), 'wb') as f:
        pickle.dump(video_original_size_dict, f, pickle.HIGHEST_PROTOCOL)
    return video_test, video_original_size



## 从图像创建视频
def create_new_video(save_path, video_name, image_array):
    (h, w) = image_array[0].shape[:2]
    if len(video_name.split('/')) > 1:
        video_name = video_name.split('/')[1]
    else:
        video_name = video_name.split('.mp4')[0]
        video_name = video_name + '.avi'
    save_video_path = os.path.join(save_path, video_name)
    output_video = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (w, h), True)
    for frame in range(len(image_array)):
        output_video.write(image_array[frame])
    output_video.release()
    cv2.destroyAllWindows()


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


## 测试采样率
def setting_sample_rate(num_frames_to_extract, sampling_rate, video, fps, ucf101_fps):
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # num_frames = int(video_length * fps)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    if num_frames_to_extract == 'all':
        sample_start_point = 0
        if fps != ucf101_fps and sampling_rate != 0:
            sampling_rate = math.ceil(fps / (ucf101_fps / sampling_rate))
    elif video_length < (num_frames_to_extract * sampling_rate):
        sample_start_point = 0
        sampling_rate = 2
    else:
        sample_start_point = sample(range(num_frames - (num_frames_to_extract * sampling_rate)), 1)[0]
    return sample_start_point, sampling_rate, num_frames


## 从视频中采样图片
def capture_and_sample_video(raw_data_dir, video_name, num_frames_to_extract, sampling_rate, fps, save_path,
                             ucf101_fps, processing_mode):
    video = cv2.VideoCapture(os.path.join(raw_data_dir, video_name))
    if fps == 'Not known':
        fps = video.get(cv2.CAP_PROP_FPS)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # ====== 设置起始采样点 ======
    sample_start_point, sampling_rate, num_frames = setting_sample_rate(num_frames_to_extract, sampling_rate, video,
                                                                        fps, ucf101_fps)
    image_array = []
    if num_frames_to_extract == 'all':
        num_frames_to_extract = int(num_frames / sampling_rate) if sampling_rate != 0 else num_frames
    if processing_mode == 'live':
        transform = set_transforms(mode='test')
    for frame in range(num_frames_to_extract):
        video.set(1, sample_start_point)
        success, image = video.read()
        if not success:
            # print('Error in reading frames from raw video')
            pass
        else:
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if processing_mode == 'live' else image
            image = Image.fromarray(RGB_img.astype('uint8'), 'RGB')
            if processing_mode == 'live':
                image_array += [transform(image)]
            else:
                image_array += [np.uint8(image)]
        sample_start_point = sample_start_point + sampling_rate
    video.release()
    if processing_mode == 'main':
        create_new_video(save_path, video_name, image_array)
    return image_array, [video_width, video_height]


if __name__ == '__main__':
    args = parser.parse_args()
    global_dir = os.path.normpath(args.row_data_dir + os.sep + os.pardir)
    ## args.row_data_dir=/hy-tmp/data/UCF-101 os.sep=/ os.pardir=.. global_dir=/hy-tmp/data/
    folder_name = '{}_sampled_data_video_sampling_rate_{}_num frames extracted_{}'.format(args.dataset,
                                                                                          args.sampling_rate,
                                                                                          args.num_frames_to_extract)
    folder_dir = os.path.join(global_dir, folder_name)
    ## folder_name=UCF101_sampled_data_video_sampling_rate_10_num frames extracted_5
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    ## 存储训练参数配置
    save_setting_info(args, "cpu", folder_dir)
    ## 处理视频
    main_procesing_data(args, folder_dir)

