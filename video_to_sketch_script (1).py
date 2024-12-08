'''
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg 
git clone https://huggingface.co/spaces/svjack/video-to-sketch && cd video-to-sketch

pip install gradio huggingface_hub torch==1.11.0 torchvision==0.12.0 pytorchvideo==0.1.5 pyav==11.4.1

huggingface-cli download \
  --repo-type dataset svjack/video-dataset-Lily-Bikini-organized \
  --local-dir video-dataset-Lily-Bikini-organized

python video_to_sketch_script.py video-dataset-Lily-Bikini-organized video-dataset-Lily-Bikini-sketch-organized --copy_others
'''

import gc
import os
import shutil
import argparse
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL.Image import Resampling
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.io import write_video
from torchvision.transforms.functional import resize
from tqdm import tqdm

from modeling import Generator

MAX_DURATION = 60
OUT_FPS = 18
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"

# Load the model
model = Generator(3, 1, 3)
weights_path = hf_hub_download("nateraw/image-2-line-drawing", "pytorch_model.bin")
model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
model.eval()

def process_one_second(vid, start_sec, out_fps):
    """Process one second of a video at a given fps
    Args:
        vid (_type_): A pytorchvideo.EncodedVideo instance containing the video to process
        start_sec (_type_): The second to start processing at
        out_fps (_type_): The fps to output the video at
    Returns:
        np.array: The processed video as a numpy array with shape (T, H, W, C)
    """
    # C, T, H, W
    video_arr = vid.get_clip(start_sec, start_sec + 1)["video"]
    # C, T, H, W where T == frames per second
    x = uniform_temporal_subsample(video_arr, out_fps)
    # C, T, H, W where H has been scaled to 256 (This will probably be no bueno on vertical vids but whatever)
    x = resize(x, 256, Resampling.BICUBIC)
    # C, T, H, W -> T, C, H, W (basically T acts as batch size now)
    x = x.permute(1, 0, 2, 3)

    with torch.no_grad():
        # T, 1, H, W
        out = model(x)

    # T, C, H, W -> T, H, W, C Rescaled to 0-255
    out = out.permute(0, 2, 3, 1).clip(0, 1) * 255
    # Greyscale -> RGB
    out = out.repeat(1, 1, 1, 3)
    return out

def process_video(input_video_path, output_video_path):
    start_sec = 0
    vid = EncodedVideo.from_path(input_video_path)
    duration = min(MAX_DURATION, int(vid.duration))
    for i in tqdm(range(duration), desc="Processing video"):
        video = process_one_second(vid, start_sec=i + start_sec, out_fps=OUT_FPS)
        gc.collect()
        if i == 0:
            video_all = video
        else:
            video_all = np.concatenate((video_all, video))

    write_video(output_video_path, video_all, fps=OUT_FPS)

def copy_non_video_files(input_path, output_path):
    """Copy non-video files and directories from input path to output path."""
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if not item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            dest_path = os.path.join(output_path, item)
            if os.path.isdir(item_path):
                shutil.copytree(item_path, dest_path)
            else:
                shutil.copy2(item_path, dest_path)

def main(input_path, output_path, copy_others=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if copy_others:
        copy_non_video_files(input_path, output_path)

    for video_name in os.listdir(input_path):
        if video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_video_path = os.path.join(input_path, video_name)
            output_video_path = os.path.join(output_path, video_name)
            process_video(input_video_path, output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to convert them into sketch videos.")
    parser.add_argument("input_path", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("output_path", type=str, help="Path to the output directory for processed videos.")
    parser.add_argument("--copy_others", action="store_true", help="Copy non-video files and directories from input to output.")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.copy_others)