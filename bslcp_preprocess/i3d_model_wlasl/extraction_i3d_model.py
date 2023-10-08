import torch
from pytorch_i3d import InceptionI3d  # Import your I3D model architecture
from torchvision import transforms
from PIL import Image
from test_i3d import load_rgb_frames_from_video


# Instantiate the model
model = InceptionI3d(in_channels=3)

# Load pre-trained weights
# checkpoint = torch.load('./weights/rgb_imagenet.pt')
checkpoint = torch.load('./weights/checkpoint_050.pth.tar', map_location=torch.device('cpu'))


# Extract and load the state dictionary
model_dict = model.state_dict()  # Get the initial model state dictionary
pretrained_checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # Filter out unnecessary keys

model_dict.update(pretrained_checkpoint_dict)  # Update the model state dictionary
model.load_state_dict(model_dict)  # Load the updated state dictionary

frames = load_rgb_frames_from_video("./05727.mp4", 0, num=-1)

print("len frames: ", len(frames))

print("Original size:", frames.size())

# Reshape the tensor
# Add batch size dimension and permute dimensions
reshaped_frames = frames.unsqueeze(0).permute(0, 4, 1, 2, 3)



# print(frames.size())

with torch.no_grad():
    # spatial_features = model(reshaped_frames)
    features = model.extract_features(reshaped_frames)
    # torch.set_printoptions(precision=6, sci_mode=False)
    spatial_features = features[0, :, :, 0, 0].T
    print(spatial_features)
    print(spatial_features.shape)
    # for frame_index in range(spatial_features.shape[2]):
    #     # frame_features = spatial_features[:, :, frame_index, :, :]
    #     frame_features = spatial_features[0, :, frame_index, 0, 0].view(1, -1)  # Assuming batch size is 1
    #     features = spatial_features[0, 0, frame_index, 0, :].view(1, -1)  # Assuming batch size is 1
    #     # Now you can process or analyze the `frame_features` tensor for each frame
    #     print(f"Frame index: {frame_index}")
    #     print(frame_features)
    #     print(frame_features.shape)
    #     print(features)
    #     print(features.shape)
        # print("=" * 20)

    




