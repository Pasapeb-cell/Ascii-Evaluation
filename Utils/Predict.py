import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image


def extract_patches(image, n, m):  
    patches = []
    image = np.array(image)
    height, width = image.shape[:2]
    images_height =0
    images_width = 0

    for y in range(0, height - n + 1, m):
        images_height += 1
        for x in range(0, width - n + 1, m):  
            images_width += 1          
            patch = image[y:y+n, x:x+n]
            patches.append(patch)
    images_width = images_width//images_height

    return torch.stack([T.ToTensor()(patch) for patch in patches]), images_width, images_height


transforms = T.Compose([T.Resize(size=(64,64)),
                        T.Normalize((0.5,), (0.5,))])
import matplotlib.pyplot as plt
def preprocess_image(image,patch_size=16, stride=4,batch_size=64):
    img = Image.fromarray(image).convert("L")
    original_size = img.size
    img = T.Grayscale(num_output_channels=1)(img)
    img_patches,images_char_width, images_char_height = extract_patches(img , patch_size, stride)
    img_patches = transforms(img_patches)
    data = DataLoader(img_patches, batch_size=batch_size, shuffle=False)
    return data , images_char_width, images_char_height ,original_size


def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        out = []
        for batch in dataloader:            
            o = model(batch.to(model.device))
            o = torch.argmax(o, axis=1)
            for i in o:
                out.append(int(i))
    return out
        
def postprocess(predictions, images_width, images_height,dictionary):
    predictions = [dictionary[i] for i in predictions]
    text = ""
    for i in range(images_width*images_height):
        if i % images_width == 0:
            text += "\n"
        text += predictions[i]
    text = text.replace('$','\\$')
    return text
