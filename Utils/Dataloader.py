import numpy as np
import random
from  torch.utils.data import DataLoader , Dataset
from .TextToImage import TextToImage

class AsciiDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
    def __len__(self):
        return self.df.shape[0]-1
    
    def __getitem__(self, idx):
        
        return  self.df['Text'][idx]
        
class CaptionDataset(AsciiDataset):
    def __init__(self, dataframe,transforms=None):
        super().__init__(dataframe)
        self.transforms = transforms
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        if idx >= len(self): raise IndexError

        text = super().__getitem__(idx)
        img = TextToImage(text)
        if self.transforms:
            img = self.transforms(img)
        return {
            "image": img,
            "text": text,
            "height": img.shape[1],
            "width": img.shape[2],
            "caption_id": idx,
            "image_id": idx            
        }

class CropDataset(AsciiDataset):
    def __init__(self, dataframe,dictionary,transforms=None,kernel_size=9):
        super().__init__(dataframe)
        self.transforms = transforms
        self.kernel_size = kernel_size
        self.dictionary = dictionary
    
    def __getitem__(self, idx):
        text = super().__getitem__(idx)

        
        def random_square_section(text, kernel_size):
            lines = text.split('\n')
            
            max_start_row = len(lines) - kernel_size
            max_start_col = len(lines[0]) - kernel_size
            
            start_row = random.randint(0, max_start_row)
            start_col = random.randint(0, max_start_col)
            
            square_section = [line[start_col:start_col+kernel_size] for line in lines[start_row:start_row+kernel_size]]
            center = kernel_size//2
            # print(idx,center)
            center_element = square_section[center][center]

            return '\n'.join(square_section), center_element

        try:
            m ,l= random_square_section(text, self.kernel_size)
        except: 
            return self.__getitem__(random.randint(0,len(self)))      
          
        img = TextToImage(m)
        if self.transforms:
            img = self.transforms(img)
        # print(l,idx )
        return img , m, l, self.dictionary[l]