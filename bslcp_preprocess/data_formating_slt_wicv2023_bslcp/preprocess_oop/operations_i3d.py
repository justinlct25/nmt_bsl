import cv2
import torch
from i3d_wlasl.pytorch_i3d import InceptionI3d  # Import I3D model architecture
from i3d_wlasl.test_i3d import load_rgb_frames_from_video


class I3dOperator():

    def __init__(self, weights_file='./weights/checkpoint_050.pth.tar'):
        self.weights_path = weights_file
        self.model = InceptionI3d(in_channels=3)
        self.model_init(weights_file)
        self.fps = None

    def model_init(self, weights_path):
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model_dict = self.model.state_dict()  # Get the initial model state dictionary
        pretrained_checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # Filter out unnecessary keys
        model_dict.update(pretrained_checkpoint_dict)  # Update the model state dictionary
        self.model.load_state_dict(model_dict)  # Load the updated state dictionary
    
    def load_video_file(self, video_file):
        self.video_file = video_file
        self.video_cap = cv2.VideoCapture(video_file)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
    
    def ms_to_frame(self, ms_list):
        result = []
        for ms in ms_list:
            result.append(self.fps * (ms/1000))
        return result

    def extract_sign_features(self, start_frame, end_frame):
        frames_num = int(end_frame - start_frame)
        frames = load_rgb_frames_from_video(self.video_file, start=start_frame, num=frames_num)
        frames = frames.unsqueeze(0).permute(0, 4, 1, 2, 3)
        with torch.no_grad():
            features = self.model.extract_features(frames)
            spatial_features = features[0, :, :, 0, 0].T
            return spatial_features

