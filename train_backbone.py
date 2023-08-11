import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import numpy as np

import time

import cv2



import torch.utils.data as Data
#from utils import progress_bar

lr = 0.01

# def adjust_learning_rate(epoch):
#     if (epoch < 50):
#         lr = 0.1
#     if (epoch >=50 and epoch <100):
#         lr = 1e-2
#     if (epoch >=100):
#         lr = 1e-3
#     return lr
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


image = np.zeros([50000,3,32,32], dtype = np.float32)
image_test = np.zeros([10000,3,32,32], dtype = np.float32)
time_start = time.time()
for i in range (50000):
    
    img = cv2.imread("../cifar_10_images/train_cifar10/%d.jpg"%(i))
    #img1 = img
    #img = img.reshape(3,32,32)
    #img = np.transpose(img, (2, 0, 1))
    #img = Image.open("../cifar_10_images/train_cifar10/%d.jpg"%(i)).convert('RGB')
    #img = 
    img = np.transpose(img, (2, 0, 1))
    image[i] = img
    
for i in range (10000):
    img = cv2.imread("../cifar_10_images/test_cifar10/%d.jpg"%(i))
    #img = np.transpose(img, (2, 0, 1))
    #img = Image.open("../cifar_10_images/test_cifar10/%d.jpg"%(i)).convert('RGB')
    img = np.transpose(img, (2, 0, 1))
    image_test[i] = img
    
    
time_end = time.time()
time_total  = time_end - time_start
print("the total decoding time is:%.6f"%(time_total))

train_label = np.load("cifar_10_labels.npy")
test_label = np.load("cifar_10_labels_test.npy")

image_tensor=torch.from_numpy(image.astype(np.float32))
image_test_tensor = torch.from_numpy(image_test.astype(np.float32))
training_data = torch.zeros([50000,3,32,32])
testing_data = torch.zeros([10000,3,32,32])
for i in range (50000):
    training_data[i] = transform_train(image_tensor[i])
for i in range (10000):
    testing_data[i] = transform_test(image_test_tensor[i])
#test_data = transform_test(image_test)
train_label_tensor=torch.from_numpy(train_label.astype(np.int64))
test_label_tensor=torch.from_numpy(test_label.astype(np.int64))
train_data=Data.TensorDataset(training_data,train_label_tensor)
test_data=Data.TensorDataset(testing_data,test_label_tensor)

trainloader = torch.utils.data.DataLoader(
train_data, batch_size=128, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(
test_data, batch_size=100, shuffle=False, num_workers=8)

# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iter = iter + 1
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     #% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.8f | Acc: %.8f%% (%d/%d)'% (train_loss/(iter * 128), 100.*correct/total, correct, total))
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    iter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iter = iter + 1
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.8f | Acc: %.8f%% (%d/%d)'% (test_loss/(iter * 100), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt1.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    #lr = adjust_learning_rate(epoch)
    train(epoch)
    test(epoch)
    scheduler.step()