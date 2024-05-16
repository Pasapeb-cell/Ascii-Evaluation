import argparse
import numpy as np
import torch
import cv2
import os
from PIL import Image

def load_image(img):    

    if img is None:
        return None, None, None
    
    height, width = float(img.shape[0]), float(img.shape[1])
    if width > height:
        new_width, new_height = (512, int(512 / width * height))
    else:
        new_width, new_height = (int(512 / height * width), 512)
    img = cv2.resize(img, (new_width, new_height))

    img = preprocess(img)
    x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
    x = torch.tensor(x).float()

    return x

def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float64) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 3), dtype=np.float64)
    ret[0:h,0:w,0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred