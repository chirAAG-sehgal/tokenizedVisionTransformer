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
import numpy as np
import cv2

class DARPA_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(os.path.join(root_dir, "normal")) + os.listdir(os.path.join(root_dir, "wound"))
        self.labels = [1 if 'wound' in img else 0 for img in self.images]
        
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
    
class MajorCrop_Rescale: #cropping major poart of the gaussian image and rescaking the object to 256x256
    def __init__(self, input_size= 256):
        self.input_size = input_size
    def __call__(self, sample):
        image, label = sample[image], self[label]


    
def get_dataloader(root_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = DARPA_Dataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader



def flood_fill(image):
    coloumns, rows = image.shape
    tl_x, tl_y = 0, 0 
    bl_x, bl_y = 0, coloumns
    tr_x, tr_y = rows, 0
    tr_x, tr_y = rows, coloumns
    def is_valid(nx, ny):
        if (nx<rows) and (ny<coloumns):
            return True
        else:
            return False
    
    def infection(nx, ny):
        if not is_valid(nx, ny):
            pass
        else:
            if (list(image[ny, nx]) != [0,0,0]):
                return nx, ny
            else:
                infection(nx+1, ny+1)
                infection(nx-1, ny)
                infection(nx+1, ny)
                infection(nx, ny-1)
                infection(nx, ny+1)
    infection(tl_x, tl_y)
    image = cv2.drawCircle(image, (tl_x, tl_y), 1, (0,0,0), -1)
    cv2.imshow('image', image)

def main():
    flood_fill(cv2.imread(''))

