import torch.utils.data as data
from PIL import Image
import os
import torch

class AsciiDataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        fi = lambda x:  x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")
        self.images = list(filter(fi, os.listdir(data_path)))
        
    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_path, self.images[idx]))
        bg = Image.new("RGBA", img.size, (255, 255, 255) + (255,))
        alpha = img.convert('RGBA').split()[-1]
        bg.paste(img, mask=alpha)
        img = bg.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(idx)