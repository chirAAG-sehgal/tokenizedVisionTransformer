#initial classes to train the transformer include - wound/no wound
#severe haemmorhage is when body covered with more than 50% blood simple
#algorithm for report should be if wounds present on most of the body parts

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import os

class DARPA_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
        self.labels = [0 if 'no_wound' in img else 1 for img in self.images]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_dataloader(root_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = DARPA_Dataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

