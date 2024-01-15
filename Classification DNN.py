#%%
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


# 读取数据
train_dir='D:\清华学习\大三第2学期\媒体与认知\实验\Project\Classification\Data\Train'
val_dir='D:\清华学习\大三第2学期\媒体与认知\实验\Project\Classification\Data\Val'
transform  = transforms.Compose([
         transforms.Resize((32,32)),
         #transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.4, 0.4, 0.4]),
])#读取数据时对数据进行初步处理：大小归一化，值强度归一化
trainset = ImageFolder(train_dir, transform=transform)
valset=ImageFolder(val_dir,transform=transform)

'''
# 深度学习中图片数据一般保存成CxHxW，即通道数x图片高x图片宽
print(trainset[0][0].size())
print(valset[0][0].size())
# 0.2和0.4是标准差和均值的近似
a=transforms.ToPILImage()(trainset[0][0]*0.2+0.4)
plt.imshow(a)
plt.show()
b=transforms.ToPILImage()(valset[0][0]*0.2+0.4)
plt.imshow(b)
plt.show()
'''
#function to show an image
def imshow(ing):
    img = img / 2 + 0.5    #unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
# get some random training images
detaiter = iter(trainloader)
images, labels = dataiter.next()

#show images
imshow(torchvision.utils.make_grid(images[:4]))
#print labels
print(' ' .join('%5s' % classes[labels[j]] for j in range(4)))


#%%
#数据划分batch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                         shuffle=True, num_workers=0)
classes = ('i2', 'i4', 'i5', 'io','ip','p5','p11','p23','p26','pl5','pl30','pl40','pl50','pl60'
           'pl60','pl80','pn','pne','po','w57')

#%%
import torch.nn as nn
import torch.nn.functional as F

#网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19) #输出层，分类19类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


#from torchvision import models
#from torch import nn
#net=models.resnet18(pretrained=True)
#num_ftrs=net.fc.in_features
#net.fc=nn.Linear(num_ftrs,19)


import torch.optim as optim
#定义优化模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#所有参数都是0.001的学习率和0.9的动量
#scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1) # 学习7步后学习率乘0.1递减

'''
#小样本可用方法
ignored_params= list(map(id,net.fc.parameters())) #只对最后一层进行优化
base_params=filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer=torch.optim.SGD([
    {'params':base_params}, # 除最后一层fc之外的其余层学习率是0
    {'params':net.fc.parameters(),'lr':0.001} # fc学习率是0.001
],lr=0,momentum=0.9)
'''

#%%
import os
# 指定GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
#net=net.cuda()
#训练网络
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
#    for i, data in enumerate(trainloader, 0):

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        #inputs=inputs.cuda()
        #labels=labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step() #学习率递减

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

#%%
PATH = './1_net.pth'
torch.save(net.state_dict(), PATH)
dataiter = iter(valloader)
images, labels = dataiter.next()

#第1个batch的分类结果
# print images
imshow(torchvision.utils.make_grid(images[:4]))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

#统计正确率
correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

#%%
#统计各类别的正确率
class_correct = list(0. for i in range(19))
class_total = list(0. for i in range(19))
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += (predicted[i] == labels[i]).sum().item()
            class_total[label] += 1

for i in range(19):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
