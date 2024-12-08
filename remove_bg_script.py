'''
pip install torch accelerate opencv-python pillow numpy timm kornia prettytable typing scikit-image transformers>=4.39.1 gradio==4.44.1 gradio_imageslider loadimg>=0.1.1 "httpx[socks]" moviepy==1.0.3

huggingface-cli download \
  --repo-type dataset svjack/video-dataset-Lily-Bikini-organized \
  --local-dir video-dataset-Lily-Bikini-organized

python remove_bg_script.py video-dataset-Lily-Bikini-organized video-dataset-Lily-Bikini-rm-background-organized --copy_others
'''

from PIL import Image, ImageChops
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
from tqdm import tqdm
from uuid import uuid1
import os
import shutil
import argparse

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision('high')  # Set precision
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def remove_background(image):
    """Remove background from a single image."""
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Convert the prediction to a mask
    mask = (pred * 255).byte()  # Convert to 0-255 range
    mask_pil = transforms.ToPILImage()(mask).convert("L")
    mask_resized = mask_pil.resize(image.size, Image.LANCZOS)

    # Apply the mask to the image
    image.putalpha(mask_resized)

    return image, mask_resized

def process_video(input_video_path, output_video_path):
    """Process a video to remove the background from each frame."""
    # Load the video
    video_clip = VideoFileClip(input_video_path)

    # Process each frame
    frames = []
    for frame in tqdm(video_clip.iter_frames()):
        frame_pil = Image.fromarray(frame)
        frame_no_bg, mask_resized = remove_background(frame_pil)
        path = "{}.png".format(uuid1())
        frame_no_bg.save(path)
        frame_no_bg = Image.open(path).convert("RGBA")
        os.remove(path)

        # Convert mask_resized to RGBA mode
        mask_resized_rgba = mask_resized.convert("RGBA")

        # Apply the mask using ImageChops.multiply
        output = ImageChops.multiply(frame_no_bg, mask_resized_rgba)
        output_np = np.array(output)
        frames.append(output_np)

    # Save the processed frames as a new video
    processed_clip = ImageSequenceClip(frames, fps=video_clip.fps)
    processed_clip.write_videofile(output_video_path, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuva420p'])

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
    parser = argparse.ArgumentParser(description="Process videos to remove background.")
    parser.add_argument("input_path", type=str, help="Path to the input directory containing videos.")
    parser.add_argument("output_path", type=str, help="Path to the output directory for processed videos.")
    parser.add_argument("--copy_others", action="store_true", help="Copy non-video files and directories from input to output.")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.copy_others)
