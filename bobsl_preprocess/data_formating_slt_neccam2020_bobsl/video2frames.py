from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import os
from helpers import time_to_sec


def sentence_video_to_frames(video_file, frames_folder, start_time, end_time, fps=25, target_width=210, target_height=260):
    video_clip = VideoFileClip(video_file)
    duration = time_to_sec(end_time) - time_to_sec(start_time)
    cut_clip = video_clip.subclip(start_time, end_time) # Cut the video clip based on the specified time range
    cut_clip = cut_clip.set_fps(fps) # Set the frame rate to the desired value
    frame_times = [t for t in range(0, int(duration * fps))] # Generate frames and save them to the output folder
    if not os.path.exists(frames_folder): # not to do the frames cutting process if the frame cut folder is already exist
        os.makedirs(frames_folder)
    for t in frame_times:
        frame_path = f"{frames_folder}/images{t+1:04d}.png"  
        os.remove(frame_path) if os.path.exists(frame_path) else None
        frame = cut_clip.get_frame(t / fps)
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((target_width, target_height))
        pil_image.save(frame_path)
    video_clip.reader.close()
    cut_clip.reader.close()
    