import torch.utils.data as data
from PIL import Image
import lightning as L
import os
import torch

class CustomDataset(data.Dataset):
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
    
class CustomDataloader(L.LightningDataModule):
    def __init__(self, batch_size, data_path,num_workers=1, transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.transform = transform
        self.num_workers = num_workers
        self.setup()
        
    def __len__(self):
        return len(self.train_dataset)
    
    def setup(self, stage=None):
            self.train_dataset = CustomDataset(self.data_path, transform=self.transform)
            self.test_dataset = CustomDataset(self.data_path, transform=self.transform)
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers)
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers)
    
    
    
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    