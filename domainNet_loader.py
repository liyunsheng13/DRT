import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class DomainNetDataset(data.Dataset):
    def __init__(self, root, image_list = '', transform = None):
        self.root = root
        self.image_list = image_list
        self.transform = transform

        self.img_ids = [l.strip().split(' ')[0] for l in open(osp.join(self.root, 'image_list', self.image_list))]
        self.img_labels = [int(l.strip().split(' ')[1]) for l in open(osp.join(self.root, 'image_list', self.image_list))]  
        self.num_classes = len(np.unique(self.img_labels))
        
    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):

        name = self.img_ids[index]

        image = Image.open(osp.join(self.root, name)).convert('RGB')

        label = self.img_labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    
