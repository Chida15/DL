#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.optim as optim
from data_loader import trainloader, testloader
from unet import UNet
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(3, 1)


# In[3]:


def train_model(model, criterion, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(trainloader.dataset)
        epoch_loss = 0
        step = 0 #minibatch数
        for x, y in trainloader:# 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()#每次minibatch都要将梯度(dw,db,...)清零
            inputs = x
            labels = y
            outputs = model(inputs)#前向传播
            loss = criterion(outputs, labels)#计算损失
            loss.backward()#梯度下降,计算出梯度
            optimizer.step()#更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // trainloader.batch_size, loss.item()))

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))


# In[4]:


def train():
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, criterion, optimizer)


# In[5]:


def create_visual_anno(anno):
    """"""
    #assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno


# In[6]:

def test():
    with torch.no_grad():
        for idx, (x, _) in enumerate(testloader):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.axis('off')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            print("./result/result{}.jpg".format(idx))
            plt.savefig("./result/result{}.jpg".format(idx), dpi=100)


# In[10]:


train()


# In[11]:


test()


# In[ ]:




