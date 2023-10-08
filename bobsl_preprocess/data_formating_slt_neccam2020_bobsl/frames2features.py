import os
import torch.nn as nn
from googlenet_pytorch import GoogLeNet
from PIL import Image
import torch
from torchvision import transforms


def frame_to_features(img_path, model):
    img = Image.open(img_path)
    img = img.resize((224, 224))  
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(img).unsqueeze(0) 
    features_tensor = torch.tensor([]).unsqueeze(0)
    torch.set_printoptions(sci_mode=False) 
    with torch.no_grad():
        features_tensor = model.extract_features(img_tensor)
    pool = nn.AdaptiveAvgPool2d((1, 1)) # Apply global average pooling to the output of the last convolutional layer
    features_tensor = pool(features_tensor).view(1, -1)
    return features_tensor


def frames_to_features(frames_folder, model):
    frames_features = torch.tensor([])
    # print("Frames cut and extracting their features...")
    # print(os.listdir(frames_folder))
    for frame_img in os.listdir(frames_folder):
        img_path = os.path.join(frames_folder, frame_img)
        if os.path.isfile(img_path):
            features = frame_to_features(img_path, model)
            frames_features = torch.cat((frames_features, features), dim=0)
        else:
            print(f"{img_path} is not valid")
    return frames_features
    
