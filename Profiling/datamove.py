import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
from torch.utils.benchmark import Timer
import os
import argparse
from torchvision.models import resnet18, resnet50
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument("-nw", "--num_worker", help="num of the workers",
                    default=4)

num_worker = int(parser.parse_args().num_worker)


data_transfer_time = []
train_time = []
epoch_time = []
forward_time = []
backward_time = []

iter_num_of_one_epoch = 20019
lr = 0.001
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

device = torch.device('cuda:2')
train_set = '/export/hdd/scratch/dataset/mini_imagenet_split/train'
val_set = '/export/hdd/scratch/dataset/mini_imagenet_split/val'

print('1. Current GPU memory usage:',
      torch.cuda.memory_allocated() / (1024**2), 'MB')

# Define the data augmentation transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r - a  # free inside reserved

print(t, r, a, f)

print('1. After loading, Current GPU memory usage:',
      torch.cuda.memory_allocated() / (1024**2), 'MB')

# Load the data into the GPU
# imagenet_data = torchvision.datasets.ImageNet(train_set)

train_dataset = datasets.ImageFolder(train_set, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=False,
                                           num_workers=num_worker,
                                           pin_memory=True)
test_dataset = datasets.ImageFolder(val_set, transform=transform)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=8,
                                          pin_memory=True)

# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]

# print(f"Label: {label}")
# pined = []
# for i, batchs in enumerate(data_loader):
#     # print(i, batchs)
#     pined.append(batchs[0].to(device))
#     # print('3. Load train data into GPU, Current GPU memory usage:',
#     #       torch.cuda.memory_allocated(device) / (1024**2), 'MB')
#     torch.cuda.empty_cache()
#     pined = []
#     print("iteration:", i)
# end_time = time.time()

# net = VGG('VGG19')
net = resnet18()
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
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def load_only(epoch):
    print('load_only')
    print('\nEpoch: %d' % epoch)
    # net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
    #       (train_loss / (iter * 128), 100. * correct / total, correct, total))
    print('iteration times', iter)


def train(epoch):
    print('train')

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    ts_epoch = time.time()
    ts_dt = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # ----------------------------data transfer------------------------------------
        inputs, targets = inputs.to(device), targets.to(device)
        te_dt = time.time()
        data_transfer_time.append(te_dt - ts_dt)
        # ----------------------------train----------------------------------------------
        ts_train = time.time()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        iter += 1
        te_train = time.time()
        train_time.append(te_train - ts_train)
        # ----------------------------------------------------------------------------------
        ts_dt = time.time()
        # -------------------------------------------------------------------------------
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    te_epoch = time.time()
    epoch_time.append(te_epoch - ts_epoch)
    # print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
    #       (train_loss / (iter * 128), 100. * correct / total, correct, total))
    print('iteration times', iter)


def train_with_fake_inr(epoch):
    print('train_with_fake_inr')
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    print('1. After loading, Current GPU memory usage:',
          torch.cuda.memory_allocated() / (1024**2), 'MB')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        break
    print('1. After loading, Current GPU memory usage:',
          torch.cuda.memory_allocated() / (1024**2), 'MB')

    for batch_idx in range(iter_num_of_one_epoch):
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
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
          (train_loss / (iter * 128), 100. * correct / total, correct, total))
    print('iteration times', iter)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    iter = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iter = iter + 1
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
          (test_loss / (iter * 100), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
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


for epoch in range(start_epoch, start_epoch + 10):
    #lr = adjust_learning_rate(epoch)
    # start.record()
    train(epoch)
    # train_with_fake_inr(epoch)
    # load_only(epoch)
    # end.record()
    # torch.cuda.synchronize()
    # epoch_time.append(start.elapsed_time(end) / 1000)  # milliseconds
    # print("the %d epoch of time is %f min" % (epoch, (t2 - t1) / 60.0))
    # test(epoch)
    # scheduler.step()
    print(
        "Epoch %d --------------------------------------------------------------------"
        % epoch)
    # print("The maximum memory usage is %d" % torch.cuda.max_memory_allocated())
    print("The sum time of all epoch", sum(epoch_time))
    print("The sum time of data_transfer time", sum(data_transfer_time))
    print("The sum time of train time", sum(train_time))
    # print("time for forward is"+str(sum(forward_time)))
    # print("time for backward is"+str(sum(backward_time)))
