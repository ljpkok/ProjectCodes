#coding:utf8
from random import sample
import argparse
import pickle
import torch
import os
from utils_action_recognition import set_project_folder_dir, \
    save_setting_info,test_model_continues_movie_youtube, load_test_data, print_error_preprocessing_movie_mode
from LRCN.create_dataset import UCF101Dataset
from torch.utils.data import DataLoader
from LRCN.lrcn_model import ConvLstm
from preprocessing_data.preprocessing_data import main_procesing_data

ucf_labels = {0: 'ApplyEyeMakeup', 1: 'ApplyLipstick', 2: 'Archery', 3: 'BabyCrawling', 4: 'BalanceBeam', 5: 'BandMarching', 6: 'BaseballPitch', 7: 'Basketball', 8: 'BasketballDunk', 9: 'BenchPress', 10: 'Biking', 11: 'Billiards', 12: 'BlowDryHair', 13: 'BlowingCandles', 14: 'BodyWeightSquats', 15: 'Bowling', 16: 'BoxingPunchingBag', 17: 'BoxingSpeedBag', 18: 'BreastStroke', 19: 'BrushingTeeth', 20: 'CleanAndJerk', 21: 'CliffDiving', 22: 'CricketBowling', 23: 'CricketShot', 24: 'CuttingInKitchen', 25: 'Diving', 26: 'Drumming', 27: 'Fencing', 28: 'FieldHockeyPenalty', 29: 'FloorGymnastics', 30: 'FrisbeeCatch', 31: 'FrontCrawl', 32: 'GolfSwing', 33: 'Haircut', 34: 'Hammering', 35: 'HammerThrow', 36: 'HandstandPushups', 37: 'HandstandWalking', 38: 'HeadMassage', 39: 'HighJump', 40: 'HorseRace', 41: 'HorseRiding', 42: 'HulaHoop', 43: 'IceDancing', 44: 'JavelinThrow', 45: 'JugglingBalls', 46: 'JumpingJack', 47: 'JumpRope', 48: 'Kayaking', 49: 'Knitting', 50: 'LongJump', 51: 'Lunges', 52: 'MilitaryParade', 53: 'Mixing', 54: 'MoppingFloor', 55: 'Nunchucks', 56: 'ParallelBars', 57: 'PizzaTossing', 58: 'PlayingCello', 59: 'PlayingDaf', 60: 'PlayingDhol', 61: 'PlayingFlute', 62: 'PlayingGuitar', 63: 'PlayingPiano', 64: 'PlayingSitar', 65: 'PlayingTabla', 66: 'PlayingViolin', 67: 'PoleVault', 68: 'PommelHorse', 69: 'PullUps', 70: 'Punch', 71: 'PushUps', 72: 'Rafting', 73: 'RockClimbingIndoor', 74: 'RopeClimbing', 75: 'Rowing', 76: 'SalsaSpin', 77: 'ShavingBeard', 78: 'Shotput', 79: 'SkateBoarding', 80: 'Skiing', 81: 'Skijet', 82: 'SkyDiving', 83: 'SoccerJuggling', 84: 'SoccerPenalty', 85: 'StillRings', 86: 'SumoWrestling', 87: 'Surfing', 88: 'Swing', 89: 'TableTennisShot', 90: 'TaiChi', 91: 'TennisSwing', 92: 'ThrowDiscus', 93: 'TrampolineJumping', 94: 'Typing', 95: 'UnevenBars', 96: 'VolleyballSpiking', 97: 'WalkingWithDog', 98: 'WallPushups', 99: 'WritingOnBoard', 100: 'YoYo'}
parser = argparse.ArgumentParser(description='UCF101 Action Recognition, LRCN architecture')
parser.add_argument('--model_dir', default='20211128-194448/Saved_model_checkpoints', type=str, help='The dir of the model we want to test')
parser.add_argument('--model_name', default='epoch_100.pth.tar', type=str, help='the name for the model we want to test on')
parser.add_argument('--video_file_name', default='a1.mp4', type=str,
                    help='the video file name we would process, if none the script would run on all of the video files in the folder')
parser.add_argument('--preprocessing_movie_mode', default='live', type=str,
                    help='should we preprocess the video on the go (live) or using the preprocessed script (default:live, options: live/preprocessed)')
parser.add_argument('--dataset', default='youtube', type=str,
                    help='the dataset name. options = youtube, UCF101')
parser.add_argument('--sampling_rate', default=10, type=int, help='what was the sampling rate of the ucf-101 train dataset')
parser.add_argument('--ucf101_fps', default=25, type=int, help='FPS of the UCF101 dataset')
parser.add_argument('--row_data_dir', default='videos/raw_videos/', type=str,help='path to find the raw data')
parser.add_argument('--open_new_folder', default='True', type=str,
                    help='open a new folder for saving the run info, if false the info would be saved in the project '
                         'dir, if debug the info would be saved in debug folder(default:True)')
parser.add_argument('--load_all_data_to_RAM', default=False, type=bool,
                    help='load dataset directly to the RAM, for faster computation. usually use when the num of class '
                         'is small (default:False')
parser.add_argument('--latent_dim', default=512, type=int, help='The dim of the Conv FC output (default:512)')
parser.add_argument('--hidden_size', default=256, type=int,
                    help="The number of features in the LSTM hidden state (default:256)")
parser.add_argument('--lstm_layers', default=2, type=int, help='Number of recurrent layers (default:2)')
parser.add_argument('--bidirectional', default=True, type=bool, help='set the LSTM to be bidirectional (default:True)')
parser.add_argument('--number_of_classes', default=101, type=int, help='The number of classes we would train on')
parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size (default:32)')
parser.add_argument('--sampled_data_dir',
                    default=r'videos/UCF-combined',
                    type=str, help='The dir for the sampled row data')

def main():
    # ====== set the run settings ======
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folder_dir = set_project_folder_dir(args.open_new_folder, args.model_dir, use_model_folder_dir=True, mode='test_youtube_movie')
    print('The setting of the run are:\n{}\n' .format(args))
    print('The training would take place on {}\n'.format(device))
    print('The project directory is {}' .format(folder_dir))
    save_setting_info(args, device, folder_dir)

    label_decoder_dict = ucf_labels ##载入标签

    ## 载入模型
    print('Loading model...')
    num_class = len(label_decoder_dict) if args.number_of_classes is None else args.number_of_classes
    model = ConvLstm(args.latent_dim, args.hidden_size, args.lstm_layers, args.bidirectional, num_class)
    model = model.to(device)
    checkpoint = torch.load(os.path.join(args.model_dir, args.model_name),map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # ====== 推理 ======
    if args.video_file_name is None and args.preprocessing_movie_mode != 'live': ##测试已经采样好的视频
        test_videos_names = [file_name for file_name in os.listdir(args.sampled_data_dir) if '.avi' in file_name]
    elif args.video_file_name is None:##测试原始视频
        test_videos_names = [file_name for file_name in os.listdir(args.row_data_dir)]
    else:##测试文件
        test_videos_names = [args.video_file_name]

    if args.preprocessing_movie_mode == 'preprocessed': ##测试处理好的视频，输入UCF-101 dataset格式
        print('processing preprocessed video')
        dataset = UCF101Dataset(args.sampled_data_dir, [test_videos_names], mode='test', dataset='youtube')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        test_model_continues_movie_youtube(model, dataloader, device, folder_dir, label_decoder_dict, args.batch_size,
                                           args.preprocessing_movie_mode, video_original_size=None)

    elif args.preprocessing_movie_mode == 'live':##实时测试新视频
        print('processing live video'+str(test_videos_names))
        movie_name_to_test = sample(test_videos_names, 1) ##随机取1个视频
        ## 先进行预处理，获得采样后的帧
        test_movie, video_original_size = main_procesing_data(args, folder_dir, sampled_video_file=movie_name_to_test,
                                                              processing_mode='live')

        print('num of test frames='+str(len(test_movie))) ##所有测试的帧数
        test_model_continues_movie_youtube(model, torch.stack(test_movie), device, folder_dir, label_decoder_dict,
                                           args.batch_size, args.preprocessing_movie_mode,
                                           video_original_size=video_original_size)
    else:
        print_error_preprocessing_movie_mode()

if __name__ == '__main__':
    main()

