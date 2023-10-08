import cv2
import torch
# from i3d_wlasl.pytorch_i3d import InceptionI3d  # Import I3D model architecture
from i3d_wlasl.pytorch_i3d_features_extraction import InceptionI3d  # Import I3D model architecture
from i3d_wlasl.test_i3d import load_rgb_frames_from_video

def model_init(weights_path='./weights/checkpoint_050.pth.tar'):
    model = InceptionI3d(in_channels=3)
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()  # Get the initial model state dictionary
    pretrained_checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # Filter out unnecessary keys
    model_dict.update(pretrained_checkpoint_dict)  # Update the model state dictionary
    model.load_state_dict(model_dict)  # Load the updated state dictionary
    return model

def get_video_fps(video_file):
    video_cap = cv2.VideoCapture(video_file)
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    return fps

def ms_to_frame(fps, ms_list):
    result = []
    for ms in ms_list:
        result.append(fps * (ms/1000))
    return result

def extract_sign_features(video_file, start_frame, end_frame, model=model_init(), is_features_per_frame=False, center_square_crop=False):
    frames_num = int(end_frame - start_frame)
    with torch.no_grad():
        if not is_features_per_frame:
            frames = load_rgb_frames_from_video(video_file, start=start_frame, num=frames_num)
            frames = frames.unsqueeze(0).permute(0, 4, 1, 2, 3)
            features = model.extract_features(frames)
            spatial_features = features[0, :, :, 0, 0].T
        else:
            spatial_features = torch.empty(0, 1024) 
            for i in range(frames_num):
                frame = load_rgb_frames_from_video(video_file, start=start_frame+i, num=1, center_square_crop=center_square_crop)
                frame = frame.unsqueeze(0).permute(0, 4, 1, 2, 3)
                features = model.extract_features(frame)
                frame_features = features[0, :, :, 0, 0].T
                spatial_features = torch.cat((spatial_features, frame_features), dim=0)
        return spatial_features
