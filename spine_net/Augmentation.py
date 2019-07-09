#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''数据增强'''
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import os

mask_path = './data/old_mask/'
image_path = './data/old_images/'
mask_save_path = './data/mask/'
image_save_path = './data/images/'
num = 100


# In[2]:


def RandomFilp(image, mask):
    '''随机翻转'''
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    else:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask


# In[3]:


def RandomRotation(image, mask):
    '''随机旋转'''
    angle = transforms.RandomRotation.get_params([-180, 180])
    image = tf.rotate(image, angle, resample=Image.NEAREST)
    mask = tf.rotate(mask, angle, resample=Image.NEAREST)
    return image, mask


# In[4]:


def RandomCrop(image, mask):
    '''随机裁剪'''
    if random.random() > 0.5:
        i, j, h, w  = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.5, 1.0), ratio = (1,1))
        image = tf.resized_crop(image, i, j, h, w, (396, 476))
        mask  = tf.resized_crop(mask, i, j, h, w, (396, 476))
    else:
        pad = random.randint(0, 192)
        image = tf.pad(image, pad)
        image = tf.resize(image, (396, 476))
        mask = tf.pad(mask, pad)
        mask = tf.resize(mask, (396, 476))
    return image, mask


# In[5]:


def transform(image, mask):
    
    # 旋转
    # angle是-180到180的随机数
    angle = transforms.RandomRotation.get_params([-180, 180])
    image = tf.rotate(image, angle, resample=Image.NEAREST)
    mask = tf.rotate(mask, angle, resample=Image.NEAREST)
    
    # 随机翻转
    if random.random() > 0.5:
        image, mask = RandomFilp(image, mask)
    
    # 随机裁剪
    if random.random() > 0.5:
        image, mask = RandomCrop(image, mask)
        
    # 随机旋转
    if random.random() > 0.5:
        image, mask = RandomRotation(image, mask)
    return image, mask


# In[6]:


def Augmentation():
    '''数据增强'''
    images_name = os.listdir(image_path)
    j = 1;
    for image_name in images_name:
        for i in range(num):
            image = Image.open(image_path + image_name)
            mask = Image.open(mask_path + image_name)
            new_image, new_mask = transform(image, mask)
            new_image.save(image_save_path + str(j) + '.png')
            new_mask.save(mask_save_path + str(j) + '.png')
            j += 1


# In[7]:


if __name__ == '__main__':
    Augmentation()


# In[ ]:





# In[ ]:




