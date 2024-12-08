---
license: other
license_name: bria-rmbg-2.0
license_link: https://bria.ai/bria-huggingface-model-license-agreement/
pipeline_tag: image-segmentation
tags:
- remove background
- background
- background-removal
- Pytorch
- vision
- legal liability
- transformers
---

# BRIA Background Removal v2.0 Model Card

RMBG v2.0 is our new state-of-the-art background removal model, designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. 
The accuracy, efficiency, and versatility currently rival leading source-available models. 
It is ideal where content safety, legally licensed datasets, and bias mitigation are paramount. 

Developed by BRIA AI, RMBG v2.0 is available as a source-available model for non-commercial use. 

## Installation and Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Update and Install Required Packages**
   ```bash
   sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg
   ```

2. **Set Up Conda Environment**
   ```bash
   conda create -n py310 python=3.10 && conda activate py310 && pip install ipykernel && python -m ipykernel install --user --name py310 --display-name "py310"
   ```

### Clone and Install the Project

1. **Clone the Repository**
   ```bash
   git clone https://huggingface.co/spaces/svjack/BRIA-RMBG-2.0-Video && cd BRIA-RMBG-2.0-Video
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Application

- To start the application, run the following command:

```bash
python app.py
```
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-start;">
        <video controls autoplay src="https://github.com/user-attachments/assets/6dc3ed5b-87a1-4cf1-9fe6-f9139cc5a83f" style="width: 100%;"></video>
    </div>
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-end;">
        <video controls autoplay src="https://github.com/user-attachments/assets/57dc6615-894d-4e00-bd12-4dcafcc3bd3f" style="width: 100%;"></video>
    </div>
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-start;">
        <video controls autoplay src="https://github.com/user-attachments/assets/2acedebd-9443-4a75-bdbf-25294d26ac8e" style="width: 100%;"></video>
    </div>
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-end;">
        <video controls autoplay src="https://github.com/user-attachments/assets/436a5acc-63bd-49b0-ab2a-868a2f706c6f" style="width: 100%;"></video>
    </div>
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-start;">
        <video controls autoplay src="https://github.com/user-attachments/assets/c02ca48b-a607-43cc-b4d3-31b7341d5854" style="width: 100%;"></video>
    </div>
    <div style="flex: 1 1 45%; margin-bottom: 10px; display: flex; justify-content: flex-end;">
        <video controls autoplay src="https://github.com/user-attachments/assets/791df32e-5e23-45f4-ab1f-72926f188b6b" style="width: 100%;"></video>
    </div>
</div>

<!--
https://github.com/user-attachments/assets/6dc3ed5b-87a1-4cf1-9fe6-f9139cc5a83f

https://github.com/user-attachments/assets/57dc6615-894d-4e00-bd12-4dcafcc3bd3f

https://github.com/user-attachments/assets/2acedebd-9443-4a75-bdbf-25294d26ac8e

https://github.com/user-attachments/assets/436a5acc-63bd-49b0-ab2a-868a2f706c6f

https://github.com/user-attachments/assets/c02ca48b-a607-43cc-b4d3-31b7341d5854

https://github.com/user-attachments/assets/791df32e-5e23-45f4-ab1f-72926f188b6b
-->

- Reomve background video in a dir
```bash
pip install torch accelerate opencv-python pillow numpy timm kornia prettytable typing scikit-image transformers>=4.39.1 gradio==4.44.1 gradio_imageslider loadimg>=0.1.1 "httpx[socks]" moviepy==1.0.3

huggingface-cli download \
  --repo-type dataset svjack/video-dataset-Lily-Bikini-organized \
  --local-dir video-dataset-Lily-Bikini-organized

python remove_bg_script.py video-dataset-Lily-Bikini-organized video-dataset-Lily-Bikini-rm-background-organized --copy_others
```
- Video to Sketch in a dir
```bash
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg 
git clone https://huggingface.co/spaces/svjack/video-to-sketch && cd video-to-sketch
pip install gradio huggingface_hub torch==1.11.0 torchvision==0.12.0 pytorchvideo==0.1.5 pyav==11.4.1
huggingface-cli download \
  --repo-type dataset svjack/video-dataset-Lily-Bikini-organized \
  --local-dir video-dataset-Lily-Bikini-organized
python video_to_sketch_script.py video-dataset-Lily-Bikini-organized video-dataset-Lily-Bikini-sketch-organized --copy_others
```

## Model Details
#####
### Model Description

- **Developed by:** [BRIA AI](https://bria.ai/)
- **Model type:** Background Removal 
- **License:** [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/)
  - The model is released under a Creative Commons license for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA. [Contact Us](https://bria.ai/contact-us) for more information. 

- **Model Description:** BRIA RMBG-2.0 is a dichotomous image segmentation model trained exclusively on a professional-grade dataset.
- **BRIA:** Resources for more information: [BRIA AI](https://bria.ai/)



## Training data
Bria-RMBG model was trained with over 15,000 high-quality, high-resolution, manually labeled (pixel-wise accuracy), fully licensed images.
Our benchmark included balanced gender, balanced ethnicity, and people with different types of disabilities.
For clarity, we provide our data distribution according to different categories, demonstrating our model’s versatility.

### Distribution of images:

| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Objects only | 45.11% |
| People with objects/animals | 25.24% |
| People only | 17.35% |
| people/objects/animals with text | 8.52% |
| Text only | 2.52% |
| Animals only | 1.89% |

| Category | Distribution |
| -----------------------------------| -----------------------------------------:|
| Photorealistic | 87.70% |
| Non-Photorealistic | 12.30% |


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Non Solid Background | 52.05% |
| Solid Background | 47.95% 


| Category | Distribution |
| -----------------------------------| -----------------------------------:|
| Single main foreground object | 51.42% |
| Multiple objects in the foreground | 48.58% |


### Architecture
RMBG-2.0 is developed on the [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) architecture enhanced with our proprietary dataset and training scheme. This training data significantly improves the model’s accuracy and effectiveness for background-removal task.<br>
If you use this model in your research, please cite:

```
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

#### Requirements
```bash
torch
torchvision
pillow
kornia
transformers
```

### Usage

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->


```python
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
torch.set_float32_matmul_precision(['high', 'highest'][0])
model.to('cuda')
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(input_image_path)
input_images = transform_image(image).unsqueeze(0).to('cuda')

# Prediction
with torch.no_grad():
    preds = model(input_images)[-1].sigmoid().cpu()
pred = preds[0].squeeze()
pred_pil = transforms.ToPILImage()(pred)
mask = pred_pil.resize(image.size)
image.putalpha(mask)

image.save("no_bg_image.png")
```

