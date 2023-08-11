import os
from PIL import Image
import re
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.models import *
import numpy as np
import time
import cv2
from PIL import Image
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-nw", "--num_worker", help="num of the workers",
                    default=4)

num_worker = int(parser.parse_args().num_worker)

#imgDir = '/export/hdd/scratch/dataset/mini_imagenet_split/train'
imgDir = '/export/hdd/scratch/dataset/102flowers/train/'
test_imgDir = '/export/hdd/scratch/dataset/102flowers/val'
batch_size = 64
batch_size_test = 100
train_nSamples = 1020
test_nSamples = 1020
init_width = 224
init_height = 224

decode_time = []
data_transfer_time = []
train_time = []
epoch_time = []


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def load_image_path(imgDir):

    all_training_files = os.walk(imgDir)
    train_files = []
    train_imageNames = []
    train_nSamples = 0
    for path, direction, filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(
            os.path.join(path, file))]
        imageNames = [file.split('.')[0]
                      for file in files if is_image_file(file)]
        files = [os.path.join(path, file)
                 for file in files if is_image_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples = train_nSamples+len(files)
    train_files = sum(train_files, [])
    train_imageNames = sum(train_imageNames, [])
    # print(train_imageNames)
    #train_imageNames.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
    #train_imageNames.sort(key = lambda x: int(x[:-4]))
    train_image_path = []
    for i in range(len(train_imageNames)):
        string = imgDir + '/' + train_imageNames[i] + '.JPEG'
        train_image_path.append(string)
    return train_image_path, train_files


def load_image_path_test(imgDir):

    all_training_files = os.walk(imgDir)
    train_files = []
    train_imageNames = []
    train_nSamples = 0
    for path, direction, filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(
            os.path.join(path, file))]
        imageNames = [file.split('.')[0]
                      for file in files if is_image_file(file)]
        files = [os.path.join(path, file)
                 for file in files if is_image_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples = train_nSamples+len(files)
    train_files = sum(train_files, [])
    train_imageNames = sum(train_imageNames, [])
    # print(train_imageNames)
    #train_imageNames.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
    #train_imageNames.sort(key = lambda x: int(x[:-4]))
    train_image_path = []
    for i in range(len(train_imageNames)):
        string = imgDir + '/' + train_imageNames[i] + '.JPEG'
        train_image_path.append(string)
    return train_image_path, train_files


train_files, train_image_path = load_image_path(imgDir)

test_files, test_image_path = load_image_path_test(test_imgDir)
train_label = np.load(
    "/export/hdd/scratch/dataset/102flowers/labels/train_label.npy")
train_label = train_label - 1
test_label = np.load(
    "/export/hdd/scratch/dataset/102flowers/labels/val_label.npy")
# print(min(test_label))
test_label = test_label - 1


class listDataset(Dataset):
    def __init__(self, files_root, target, nsamples, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=0):

        self.image_root = files_root
        self.target = target
        self.nSamples = nsamples
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        imgpath = self.image_root[index]
        img = Image.open(imgpath).convert('RGB')
     # print(img)
        if self.shape is not None:
            img = img.resize(self.shape)
        if self.transform is not None:
            img = self.transform(img)
        label = self.target[index]
        # print(label.type)
        label = torch.from_numpy(np.array(label, dtype=np.int64))
        return (img, label)


lr = 0.01
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomCrop(224),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.44501326),
                         (0.24577609, 0.24310623, 0.26083428))
])


train_loader = torch.utils.data.DataLoader(
    listDataset(train_image_path, train_label, train_nSamples, shape=(init_width, init_height),
                shuffle=False,
                transform=transform_train,
                train=True,
                seen=0,
                batch_size=batch_size,
                num_workers=0),
    batch_size=batch_size, shuffle=True, drop_last=True,  num_workers=num_worker)


test_loader = torch.utils.data.DataLoader(
    listDataset(test_image_path, test_label, test_nSamples, shape=(init_width, init_height),
                shuffle=False,
                transform=transform_test,
                train=False,
                seen=0,
                batch_size=batch_size,
                num_workers=0),
    batch_size=batch_size_test, shuffle=False, num_workers=8)


#net = ResNet18()
net = torchvision.models.resnet18(pretrained=True)
# net=torchvision.models.resnet18()
fc_features = net.fc.in_features
net.fc = nn.Linear(fc_features, 102)
# print(net)
net = net.to(device)
# if device == 'cuda':
#net = torch.nn.DataParallel(net)
#cudnn.benchmark = True

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

        iter = iter + 1
        te_train = time.time()
        train_time.append(te_train - ts_train)
        # ----------------------------------------------------------------------------------
        ts_dt = time.time()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
    #       (train_loss/(iter * 128), 100.*correct/total, correct, total))
    te_epoch = time.time()
    epoch_time.append(te_epoch - ts_epoch)


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
          (test_loss/(iter * 100), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt_recon_miniimagenet_raw_30.pth')
    #     best_acc = acc


for epoch in range(start_epoch, start_epoch+10):
    #lr = adjust_learning_rate(epoch)
    start_time = time.time()
    train(epoch)
    # test(epoch)
    end_time = time.time()
    print(
        "Epoch %d --------------------------------------------------------------------"
        % epoch)
    # print("The maximum memory usage is %d" % torch.cuda.max_memory_allocated())
    # print("the time for this epoch is:", end_time - start_time)
    print("The sum time of all epoch", sum(epoch_time))
    print("The sum time of data_transfer time", sum(data_transfer_time))
    print("The sum time of train time", sum(train_time))
    scheduler.step()
