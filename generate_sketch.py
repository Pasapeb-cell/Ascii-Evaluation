import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel

import open_clip
import cv2
from sentence_transformers import util

import os
from torchvision import transforms as T
import pandas as pd
from Utils.Dataloader import CropDataset
import matplotlib.pyplot as plt
from Utils.TextToImage import unique_char_mapping
from Utils.Filters import applyTransforms

from PIL import Image
import numpy as np
import torch

from SketchKeras.sketch_model import SketchKeras
import Utils.sketchkeras as SK 
import Utils.Predict as P 

def sketch(image,theshold=253):
    img_shape = image.shape

    image = SK.load_image(image)

    image = sketchKeras(image.to(model_VIT.device))
    image = image.cpu().detach().numpy()
    
    sketch_image = SK.postprocess(image[0])
    sketch_image = T.ToPILImage()(sketch_image)
    sketch_image = T.Resize(size=(img_shape[0],img_shape[1]))(sketch_image)
    sketch_image= np.array(sketch_image)
    sketch_image = np.where(sketch_image > theshold, 255, 0).astype(np.uint8)
    return sketch_image

DATASET_PATH = './custom_datasets/datasets/Ascii_scraping/ascii_art_3_2.csv'
PATH_RUNS = os.environ.get("PATH_RUNS", "saved_models/")
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", os.path.join(PATH_RUNS, "checkpoints/"))
NUM_WORKERS = os.cpu_count()

df = pd.read_csv(DATASET_PATH)
df.columns =["h","w",'Text']

df['Text'] = df['Text'].apply(lambda x : x.replace('$','\\$'))
text = ' '.join(df['Text'].values)

chardict = unique_char_mapping(text)
inv_chardict = {v: k for k, v in chardict.items()}




dataset = CropDataset(df,
                      dictionary=chardict,
                      transforms=T.Compose([applyTransforms,
                                            T.ToPILImage(),
                                            T.Grayscale(num_output_channels=1),
                                            T.Resize((64, 64)),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.5], std=[0.5])
                                            ]),
                      kernel_size=3)

from VIT.Clasification import ViT
model_kwargs={
        "embed_dim": 256,
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "patch_size": 8,
        "num_channels": 1,
        "num_patches": 64,
        "num_classes": len(chardict),
        "dropout": 0.2,
        "lr": 1e-3,
    }
model_VIT = ViT.load_from_checkpoint("saved_models/checkpoints/VIT/SwaV-epoch=14-train_loss=0.00.ckpt",**model_kwargs)


SKETCH_PATH = 'SketchKeras/model.pth'
sketchKeras = SketchKeras()
sketchKeras.load_state_dict(torch.load(SKETCH_PATH))
sketchKeras.to(model_VIT.device)

image = np.array(Image.open("sit_toyosu.png").convert('RGB').resize((2048,2048)))

sketch = sketch(image)

Image.fromarray(sketch).save("sketch.png")