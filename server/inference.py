# coding:utf8
import os

import cv2
import torch
import numpy as np
from utils_action_recognition import set_project_folder_dir, test_model_continues_movie_youtube
from Preprocessing.preprocessing_data import main_processing_data2
import torchvision
import torchvision.transforms as T
from PIL import Image

ucf_labels = {0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling', 4: 'BalanceBeam',
              5: 'BandMarching', 6: 'BaseballPitch', 7: 'Basketball', 8: 'BasketballDunk', 9: 'BenchPress',
              10: 'Biking', 11: 'Billiards', 12: 'BlowDryHair', 13: 'BlowingCandles', 14: 'BodyWeightSquats',
              15: 'Bowling', 16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth',
              20: 'CleanAndJerk', 21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot', 24: 'CuttingInKitchen',
              25: 'Diving', 26: 'Drumming', 27: 'Fencing', 28: 'FieldHockeyPenalty', 29: 'FloorGymnastics',
              30: 'FrisbeeCatch', 31: 'FrontCrawl', 32: 'GolfSwing', 33: 'Haircut', 34: 'Hammering', 35: 'HammerThrow',
              36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump', 40: 'HorseRace',
              41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing', 44: 'JavelinThrow', 45: 'JugglingBalls',
              46: 'JumpingJack', 47: 'JumpRope', 48: 'Kayaking', 49: 'Knitting', 50: 'LongJump', 51: 'Lunges',
              52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks', 56: 'ParallelBars',
              57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf', 60: 'PlayingDhol', 61: 'PlayingFlute',
              62: 'PlayingGuitar', 63: 'PlayingPiano', 64: 'PlayingSitar', 65: 'PlayingTabla', 66: 'PlayingViolin',
              67: 'PoleVault', 68: 'PommelHorse', 69: 'PullUps', 70: 'Punch', 71: 'PushUps', 72: 'Rafting',
              73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing', 76: 'SalsaSpin', 77: 'ShavingBeard',
              78: 'Shotput', 79: 'SkateBoarding', 80: 'Skiing', 81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling',
              84: 'SoccerPenalty', 85: 'StillRings', 86: 'SumoWrestling', 87: 'Surfing', 88: 'Swing',
              89: 'TableTennisShot', 90: 'TaiChi', 91: 'TennisSwing', 92: 'ThrowDiscus', 93: 'TrampolineJumping',
              94: 'Typing', 95: 'UnevenBars', 96: 'VolleyballSpiking', 97: 'WalkingWithDog', 98: 'WallPushups',
              99: 'WritingOnBoard', 100: 'YoYo'}


def inference(filename, device, model, modelYOLO):
    batch_size = 16
    folder_dir = set_project_folder_dir(True, 'checkpoint', use_model_folder_dir=True,
                                        mode='temp')
    label_decoder_dict = ucf_labels

    # ====== 推理 ======
    print('processing live video' + filename)
    test_movie, video_original_size = main_processing_data2('temp', filename, 10)
    print('num of test frames=' + str(len(test_movie)))  ##所有测试的帧数
    predict_label = test_model_continues_movie_youtube(model, torch.stack(test_movie), device, folder_dir,
                                                       label_decoder_dict,
                                                       batch_size, 'live', video_original_size=video_original_size)

    count = np.bincount(predict_label)
    label = np.argmax(count)
    predicted_label = label_decoder_dict[label]
    print('Predicted label is: ' + predicted_label)

    # ====== YOLO ======

    # Preprocess the image
    image_array = []
    video = cv2.VideoCapture(os.path.join('temp', 'a1.mp4'))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_to_extract = frame_count
    sample_start_point = 0
    for frame in range(num_frames_to_extract):
        video.set(1, sample_start_point)
        success, image = video.read()
        if image is not None:
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(RGB_img.astype('uint8'), 'RGB')
            image_array.append(image)
        sample_start_point = sample_start_point + 10
    video.release()
    results = modelYOLO(image_array)
    obj = results.pandas().xyxy[0].name.drop_duplicates().values.tolist()

    print("Predicted obj: ", obj)
    return predicted_label, obj


if __name__ == '__main__':
    inference('a1.mp4', 'cpu', 'cpu', 'cpu')
