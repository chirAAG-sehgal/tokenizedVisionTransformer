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
import random

class DARPA_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        randomly_sample_normal = random.sample(os.listdir(f'{root_dir}/normal'), len(os.listdir(f'{root_dir}/trauma')))
        self.images = os.listdir('final_daataa/trauma') + randomly_sample_normal
        self.labels = [1]*len(os.listdir('final_daataa/trauma')) + [0]*len(os.listdir('final_daataa/trauma'))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels[idx]
        
        if label == 1:
            img_name = os.path.join('final_daataa/trauma', self.images[idx])
        else:
            img_name = os.path.join('final_daataa/normal', self.images[idx])
        
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.long).to(torch.device('cuda'))
        return image, label_tensor
    
    
# class MajorCrop_Rescale: #cropping major poart of the gaussian image and rescaking the object to 256x256
#     def __init__(self, input_size= 256):
#         self.input_size = input_size
#     def __call__(self, sample):
#         image, label = sample[image], self[label]

class Transforms_Llama:
    def __call__(self,image):
        image = np.array(image) / 255
        image = 2.0 * image - 1.0
        image_tensor = torch.tensor(image)
        image_tensor = torch.einsum('hwc->chw',image_tensor)
        image_input = image_tensor.float().to("cuda")
        del image,image_tensor
        return image_input

    
def get_dataloader(root_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        Transforms_Llama(),
    ])
    dataset = DARPA_Dataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


 


def bfs(image, startrow,startcol, visited):
    # Create a queue for BFS
    q = deque()
    row,col = image.shape
    max_x, min_x, max_y, min_y = 0, row, 0 ,col 
    # Mark the current node as visited and enqueue it
    visited[startrow][startcol] = True
    q.append([startrow,startcol])

    # Iterate over the queue
    while q:
        # Dequeue a vertex from queue and print it
        currentnode = q.popleft()
        currentrow,currentcol = currentnode
        color = image[currentrow, currentcol]
        if color!=0:
            if currentrow>max_x:
                max_x = currentrow
            if currentrow<min_x:
                min_x = currentrow
            if currentcol>max_y:
                max_y = currentcol
            if currentcol<min_y:
                min_y = currentcol

        # Get all adjacent vertices of the dequeued vertex
        # If an adjacent has not been visited, then mark it visited and enqueue it
        for i in range(-1,2):
            for j in range(-1,2):
                if currentrow-i>=0 and currentrow-i<row and currentcol-j>=0 and currentcol-j<col:
                    if not visited[currentrow-i][currentcol-j]:
                        visited[currentrow-i][currentcol-j] = True
                        q.append([currentrow-i,currentcol-j])
    return max_x, max_y, min_x, min_y

""" For creating dataset of cropped images """
# def main():
#     for i in (sorted(os.listdir('final_daataa/trauma'))):
#         try:
#             img = cv2.imread('final_daataa/trauma/'+i)
#             assert img is not None 
#             img_resized = cv2.cvtColor(cv2.resize(img, (img.shape[1]//8, img.shape[0]//8)), cv2.COLOR_BGR2GRAY)
#             visited= np.zeros(img_resized.shape, dtype=bool)
#             max_x, max_y, min_x, min_y = bfs(img_resized ,0,0, visited)
#             image_cropped = cv2.resize(img[8*min_x:8*max_x,8*min_y:8*max_y], (256,256))
#             cv2.imwrite('final_daataa/trauma_cropped/'+i, image_cropped)
#         except:
#             print(i)
#             continue
    