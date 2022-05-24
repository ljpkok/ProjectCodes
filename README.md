# ProjectCodes\begin{minted}{python}
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


def capture_and_sample_video(raw_data_dir, video_name, num_frames_to_extract, sampling_rate, fps, save_path,
                             ucf101_fps, processing_mode):
    video = cv2.VideoCapture(os.path.join(raw_data_dir, video_name))
    if fps == 'Not known':
        fps = video.get(cv2.CAP_PROP_FPS)
    video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
            print('Error in reading frames from raw video')
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
\end{minted}
