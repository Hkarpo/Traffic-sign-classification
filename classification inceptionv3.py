#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


# In[3]:


# 读取数据
train_dir='/home/tsinghuaee113/jupyter_projects/Classification/Data/Train'
val_dir='/home/tsinghuaee113/jupyter_projects/Classification/Data/Val'
test_dir='/home/tsinghuaee113/jupyter_projects/Classification/Data/Test'
transform  = transforms.Compose([
    transforms.Resize((299,299)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.4, 0.4, 0.4]),
])
#读取数据时对数据进行初步处理：大小归一化，值强度归一化
trainset = ImageFolder(train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

valset=ImageFolder(val_dir,transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                        shuffle=True, num_workers=4)
'''
testset=ImageFolder(test_dir,transform=transform)       #error？
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                        shuffle=True, num_workers=4)
'''
classes = ('i2', 'i4', 'i5', 'io','ip','p5','p11','p23','p26','pl5','pl30','pl40','pl50','pl60'
           'pl60','pl80','pn','pne','po','w57')


# In[4]:


import torch.nn as nn
import torch.nn.functional as F
# get model and replace the original fc layer with your fc layer
from torchvision import models
from torch import nn
net=models.inception_v3(pretrained=True)
num_ftrs=net.fc.in_features
net.fc=nn.Linear(num_ftrs,19)
net.aux_logits=False


# In[5]:


#定义优化模型
import torch.optim as optim
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.8)
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10000,gamma=0.3) # 学习10000步后学习率乘0.1递减


# In[6]:


# training part
import os

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"]="2"
net=net.cuda()

#训练网络
for epoch in range(32):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs=inputs.cuda()
        labels=labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


# In[7]:


PATH = './1_inception.pth'
torch.save(net.state_dict(), PATH)


# In[8]:


net = models.inception_v3(pretrained=True)
num_ftrs=net.fc.in_features
net.fc=nn.Linear(num_ftrs,19)

net.load_state_dict(torch.load(PATH))


# In[9]:


#统计正确率
correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


# In[10]:


#统计各类别的正确率
class_correct = list(0. for i in range(19))
class_total = list(0. for i in range(19))
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.logits, 1)
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += (predicted[i] == labels[i]).sum().item()
            class_total[label] += 1

for i in range(19):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:


#测试集结果
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.logits, 1)
#predicted是预测的label，但没看懂存成什么格式，改天再说

