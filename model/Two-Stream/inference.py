#coding:utf8
import torch
import numpy as np
from lib import slowfastnet
import cv2
import sys
import os
import argparse

## 输入模型权重，输入视频，结果图片路径，结果视频路径，帧率
parser = argparse.ArgumentParser('inference images')
parser.add_argument('--inputvideo', dest='inputvideo', help='input video', type=str, default='videos/太极拳1.mp4')
parser.add_argument('--outputvideo', dest='outputvideo', help='output video', type=str, default='videos/result_videos/太极拳1.mp4')
parser.add_argument('--outputimagesdir', dest='outputimagesdir', help='output images', type=str, default='videos/太极拳1')
parser.add_argument('--samplefrequency', dest='samplefrequency', help='video samplefrequency', type=int, default=1)
parser.add_argument('--model', dest='model', help='model', type=str, default='ResNet50')
parser.add_argument('--weightspath', dest='weightspath', help='input directory for modelweights', type=str, default='checkpoints/2021-11-20-22-23-58/clip_len_64frame_sample_rate_1_checkpoint_0.pth.tar')
args = parser.parse_args()

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('101class_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    ## 初始化模型
    if args.model == "ResNet50":
        model = slowfastnet.resnet50(class_num=101)

    checkpoint = torch.load(args.weightspath, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    torch.no_grad()

    ## 输出图像文件夹
    imageresult = args.outputimagesdir
    if not os.path.exists(imageresult):
        os.mkdir(imageresult)

    ## 读取视频
    capin = cv2.VideoCapture(args.inputvideo)
    if capin.isOpened()== False:
        print('bad video')
    else:
        width  = capin.get(cv2.CAP_PROP_FRAME_WIDTH)  
        height = capin.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = int(capin.get(cv2.CAP_PROP_FPS)) #获得帧率
        print('video width='+str(width))
        print('video height='+str(height))
        print('video fps='+str(fps))

    retaining = True

    ## 输出视频结果
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoout = cv2.VideoWriter(args.outputvideo, fourcc, fps, (int(width),int(height)))
    clip = [] ##缓冲图像序列
    frequency = args.samplefrequency ##采样频率
    i = 0
    while retaining:
        retaining, frame = capin.read()
        if not retaining and frame is None:
            continue
        i = i + 1
        if frame.shape[1] > 2000.0:
            font = 4.0
        if frame.shape[1] > 1500.0:
            font = 2.0
        elif frame.shape[1] > 1000.0:
            font = 1.0
        else:
            font = 0.6
        if i % frequency != 0:
            continue
        tmp_ = cv2.resize(frame, (171, 128))
        tmp_ = cv2.cvtColor(tmp_, cv2.COLOR_BGR2RGB)
        tmp_ = center_crop(tmp_)
        tmp = (tmp_ - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
        clip.append(tmp)

        ## 每次取64帧进行推理
        if len(clip) == 64:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (50, int(50*font)),
                        cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 255, 255), 2)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (50, int(100*font)),
                        cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 255, 255), 2)
            cv2.imwrite(os.path.join(imageresult,str(i)+'.png'),frame)
            videoout.write(frame)
            clip.pop(0)

    capin.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()









