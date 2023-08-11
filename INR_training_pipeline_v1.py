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
from PIL import Image
import torch.utils.data as Data
import re
from torch.utils.data import Dataset


num_images = 50000
hidden_dimension  = 15
weights = []
batch_size = 128
start_epoch = 0
lr = 0.01
test_nSamples = 10000
init_width = 32
init_height = 32
batch_size_test = 100
for i in range (num_images):
    INR_weights = torch.load("./weights_1/%d.pt"%(i), map_location = torch.device('cpu'))
    weights.append(INR_weights)


test_imgDir = './cifar_10_images/test_cifar10'
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

linear_0_weight = torch.zeros(num_images, 2, hidden_dimension )
linear_0_bias = torch.zeros(num_images, hidden_dimension )
linear_1_weight = torch.zeros(num_images, hidden_dimension, hidden_dimension )
linear_1_bias = torch.zeros(num_images, hidden_dimension )
linear_2_weight = torch.zeros(num_images, hidden_dimension, hidden_dimension )
linear_2_bias = torch.zeros(num_images, hidden_dimension )
linear_3_weight = torch.zeros(num_images, hidden_dimension, 3 )
linear_3_bias = torch.zeros(num_images, 3 )

for i in range (num_images):
    
    linear_0_weight[i] = weights[i]['net.0.linear.weight'].t()
    linear_0_bias[i] = weights[i]['net.0.linear.bias']
    linear_1_weight[i] = weights[i]['net.1.linear.weight'].t()
    linear_1_bias[i] = weights[i]['net.1.linear.bias']
    linear_2_weight[i] = weights[i]['net.2.linear.weight'].t()
    linear_2_bias[i] = weights[i]['net.2.linear.bias']
    linear_3_weight[i] = weights[i]['last_layer.linear.weight'].t()
    linear_3_bias[i] = weights[i]['last_layer.linear.bias']

def load_image_path_test(imgDir):

    all_training_files=os.walk(imgDir)
    train_files=[]
    train_imageNames=[]
    train_nSamples=0
    for path,direction,filelist in all_training_files:
        files = [file for file in filelist if os.path.isfile(os.path.join(path, file))]
        imageNames = [file.split('.')[0] for file in files if is_image_file(file)]
        files = [os.path.join(path, file) for file in files if is_image_file(file)]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples=train_nSamples+len(files)
    train_files=sum(train_files,[])
    train_imageNames=sum(train_imageNames,[])
    #print(train_imageNames)
    train_imageNames.sort(key = lambda i:int(re.match(r'(\d+)',i).group()))
    #train_imageNames.sort(key = lambda x: int(x[:-4]))
    train_image_path = []
    for i in range (len(train_imageNames)):
        string = imgDir + '/' + train_imageNames[i] + '.jpg'
        train_image_path.append(string)
    return train_image_path

test_image_path = load_image_path_test(test_imgDir)
class listDataset_RAM(Dataset):
    def __init__(self,data, target, nsamples,shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=0):
      
      self.data=data
      self.target=target
      self.nSamples=nsamples
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
        #imgpath = self.image_root[index]
        #img = Image.open(imgpath).convert('RGB')
     #print(img)
        img = self.data[index]
        if self.shape is not None:
            img = img.resize(self.shape)
        if self.transform is not None:
            img = self.transform(img)
        label=self.target[index]
        #print(label.type)
        label = torch.from_numpy(np.array(label, dtype = np.int64))
     
        return (img, label)

def psnr(img1, img2):
    """Calculates PSNR between two images.
    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).pow(2).mean().log10().to('cpu').item()

def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.
    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    #coordinates = torch.ones(img.shape[1:]).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

img = torch.zeros([batch_size, 3,32,32])
coordinates = torch.zeros([batch_size, img.shape[2] * img.shape[3], 2])
for i in range (batch_size):
    coordinates[i], _= to_coordinates_and_features(img[i])
    
device = 'cuda:0'
coordinates = coordinates.to(device)

linear_0_bias = linear_0_bias.view(-1,1,linear_0_bias.shape[1])
linear_1_bias = linear_1_bias.view(-1,1,linear_1_bias.shape[1])
linear_2_bias = linear_2_bias.view(-1,1,linear_2_bias.shape[1])
linear_3_bias = linear_3_bias.view(-1,1,linear_3_bias.shape[1])

# linear_0_weight = linear_0_weight.half()
# linear_0_bias = linear_0_bias.half()
# linear_1_weight = linear_1_weight.half()
# linear_1_bias = linear_1_bias.half()
# linear_2_weight = linear_2_weight.half()
# linear_2_bias = linear_2_bias.half()
# linear_3_weight = linear_3_weight.half()
# linear_3_bias = linear_3_bias.half()
# coordinates = coordinates.half()

linear_0_weight = linear_0_weight.to(device)
linear_0_bias = linear_0_bias.to(device)
linear_1_weight = linear_1_weight.to(device)
linear_1_bias = linear_1_bias.to(device)
linear_2_weight = linear_2_weight.to(device)
linear_2_bias = linear_2_bias.to(device)
linear_3_weight = linear_3_weight.to(device)
linear_3_bias = linear_3_bias.to(device)

def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.
    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.

transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_label = np.load("cifar_10_labels.npy")
test_label = np.load("cifar_10_labels_test.npy")
train_label_tensor=torch.from_numpy(train_label.astype(np.int64))
test_label_tensor=torch.from_numpy(test_label.astype(np.int64))
train_label_tensor = train_label_tensor.to(device)
test_label_tensor=test_label_tensor.to(device)

test_data = []
for i in range (len(test_image_path)):
    img = Image.open(test_image_path[i]).convert('RGB')
    test_data.append(img)
    
test_loader = torch.utils.data.DataLoader(
        listDataset_RAM(test_data, test_label, test_nSamples, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transform_test, 
                       train=False, 
                       seen=0,
                       batch_size=batch_size,
                       num_workers=0),
        batch_size=batch_size_test, shuffle=False, num_workers=8)

iteration = int(50000/128)
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

def train(epoch, coordinates, linear_0_weight, linear_0_bias,linear_1_weight, linear_1_bias,linear_2_weight,  linear_2_bias, linear_3_weight, linear_3_bias, train_label_tensor):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    index_list = torch.randperm(50000)
    linear_0_weight = linear_0_weight[index_list]
    #print(linear_0_bias.shape)
    linear_0_bias = linear_0_bias[index_list]
    linear_1_weight = linear_1_weight[index_list]
    linear_1_bias = linear_1_bias[index_list]
    linear_2_weight = linear_2_weight[index_list]
    linear_2_bias = linear_2_bias[index_list]
    linear_3_weight = linear_3_weight[index_list]
    linear_3_bias = linear_3_bias[index_list]
    train_label_tensor = train_label_tensor[index_list]
    
    for batch_idx in range (iteration):
        #inputs, targets = inputs.to(device), targets.to(device)
        
      
        
        output1 = coordinates.matmul(linear_0_weight[batch_size * batch_idx : batch_size * (batch_idx + 1)]) + linear_0_bias[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        output1 = torch.sin(30 * output1)
        output2 = output1.matmul(linear_1_weight[batch_size * batch_idx : batch_size * (batch_idx + 1)]) + linear_1_bias[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        output2 = torch.sin(30 * output2)
        output3 = output2.matmul(linear_2_weight[batch_size * batch_idx : batch_size * (batch_idx + 1)]) + linear_2_bias[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        output3 = torch.sin(30 * output3)
        output4 = output3.matmul(linear_3_weight[batch_size * batch_idx : batch_size * (batch_idx + 1)]) + linear_3_bias[batch_size * batch_idx : batch_size * (batch_idx + 1)]
        #print(output4.shape)
        output4 = output4[:,:,[2,1,0]]
        output4 = output4.reshape(batch_size, 32,32,3)
        
        output4 = clamp_image(output4)
        output4 = output4.permute(0,3,1,2)
        #print(linear_0_weight.requires_grad)
        reconstructed_augment = transform_train(output4)
        targets = train_label_tensor[ batch_size * batch_idx : batch_size * (batch_idx + 1)]
        
        optimizer.zero_grad()
        outputs = net(reconstructed_augment)
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
    best_acc = 0
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
        torch.save(state, './checkpoint/ckpt_whole_pipeline.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+250):
    #lr = adjust_learning_rate(epoch)
    
    train(epoch, coordinates, linear_0_weight, linear_0_bias,linear_1_weight, linear_1_bias,linear_2_weight,  linear_2_bias, linear_3_weight, linear_3_bias, train_label_tensor)
    test(epoch)
    
    scheduler.step()
