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
from collections import deque

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


 


def bfs(image, startrow,startcol, visited):
    # Create a queue for BFS
    q = deque()
    row,col = image.shape

    # Mark the current node as visited and enqueue it
    visited[startrow][startcol] = True
    q.append([startrow,startcol])

    # Iterate over the queue
    while q:
        # Dequeue a vertex from queue and print it
        currentnode = q.popleft()
        currentrow,currentcol = currentnode
        color = image[currentrow][currentcol]
        if color!=0:
            return (currentnode)

        # Get all adjacent vertices of the dequeued vertex
        # If an adjacent has not been visited, then mark it visited and enqueue it
        for i in range(-1,2):
            for j in range(-1,2):
                if not visited[currentrow-i][currentcol-j] and currentrow-i>=0 and currentrow-i<row and currentcol-j>=0 and currentcol-j<col:
                    visited[currentrow-i][currentcol-j] = True
                    q.append([currentrow-i,currentcol-j])


