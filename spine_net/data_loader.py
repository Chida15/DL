#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import numpy as np
import torch

img_path = './data/images/'
msk_path = './data/mask/'
# In[2]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

target_transform = transforms.ToTensor()


# In[3]:


class SpineData(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(os.listdir(img_path))
    
    def __getitem__(self, idx):
        image_name = os.listdir(img_path)[idx]
    
        image = Image.open(img_path + image_name).convert('RGB')
        image = image.resize((256, 256))
        mask = Image.open(msk_path + image_name).convert('L')
        mask = mask.resize((256, 256))
        
        # 将image和mask都转为Tensor格式
        if self.target_transform:
            mask = self.target_transform(mask)
        #print(mask.shape)
        if self.transform:
            image = self.transform(image)
        return image, mask


# In[4]:

spine_data = SpineData(transform, target_transform)
train_size = int(0.9 * len(spine_data))
test_sizee = len(spine_data) - train_size
trainset, testset = random_split(spine_data, (train_size, test_sizee))
trainloader = DataLoader(trainset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=2)
testloader = DataLoader(testset)

# In[ ]:




