import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torchvision.models import *
import numpy as np
import time
import cv2
from PIL import Image
import torch.utils.data as Data
import re
from torch.utils.data import Dataset
import random

# the INR path:
# /export/hdd/scratch/dataset/INR_weights/mini_imagenet/trail_10x28_30fq_2000ep_7000im/trail_10x28_30fq_2000ep_7000im

# 102flowers dataset INR I use is 10 layers 32 hidden dimension
# cifar 10 dataset INR I use is 5 layers 13 hidden dimension
# miniimagenet dataset INR I use is 10 layers 40 hidden dimension

mini_label = f"/export/hdd/scratch/dataset/mini_imagenet_split/labels/train_label.npy"
flo_label = f"/export/hdd/scratch/dataset/102flowers/labels/train_label.npy"
cifar_label = f"/export/hdd/scratch/dataset/cifar_10_images/labels/cifar_10_labels.npy"

mini_weight = f"/export/hdd/scratch/dataset/INR_weights/mini_imagenet/"
flo_weight = f"/export/hdd/scratch/dataset/INR_weights/102flowers/trail_10x32_30fq_5000ep_255im/"
cifar_weight = f"/export/hdd/scratch/dataset/INR_weights/cifar_10/"


parser = argparse.ArgumentParser()
parser.add_argument("-lp", "--label_path", help="The path of the label",
                    default=cifar_label)
parser.add_argument("-ld", "--logdir", help="Path to save logs",
                    default=f"./log")
parser.add_argument("-wd", "--INR_weights_dir", help="Path to save INR weights",
                    default=cifar_weight)
parser.add_argument("-imgw", "--image_w",
                    help="The width of reconstructed image", type=int, default=224)
parser.add_argument("-imgh", "--image_h",
                    help="The height of reconstructed image", type=int, default=224)
parser.add_argument("-device", "--device",
                    help="The id of the GPU", default=f"cuda:1")
parser.add_argument("-ne", "--num_epoch",
                    help="Number of iterations to train for", type=int, default=10)
parser.add_argument("-lr", "--learning_rate",
                    help="Learning rate", type=float, default=1e-3)
parser.add_argument("-se", "--seed", help="Random seed",
                    type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-lss", "--layer_size",
                    help="Layer sizes as list of ints", type=int, default=13)
parser.add_argument("-nl", "--num_layers",
                    help="Number of layers", type=int, default=5)
parser.add_argument(
    "-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial",
                    help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-bs", "--batch_size",
                    help="the batch_size of each iteration when training.", type=int, default=64)

# parse the args:
args = parser.parse_args()
label_path = args.label_path
logdir_logfiles = args.logdir
weight_dir = args.INR_weights_dir
image_w = args.image_w
image_h = args.image_h
device = args.device
num_epoch = args.num_epoch
lr = args.learning_rate
layer_size = args.layer_size
num_layers = args.num_layers
w0 = args.w0
w0_initial = args.w0_initial
batch_size = args.batch_size

wt_path = os.listdir(weight_dir)
num_images = len(wt_path)

# time consumed variables
disk_to_gpu_time = []
decode_aug_time = []
train_time = []
forward_time = []
backward_time = []


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


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
    train_imageNames.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
    #train_imageNames.sort(key = lambda x: int(x[:-4]))
    train_image_path = []
    for i in range(len(train_imageNames)):
        string = imgDir + '/' + train_imageNames[i] + '.jpg'
        train_image_path.append(string)
    return train_image_path


# test_image_path = load_image_path_test(test_imgDir)


class listDataset_RAM(Dataset):
    def __init__(self, data, target, nsamples, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=0):

        self.data = data
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
        #imgpath = self.image_root[index]
        #img = Image.open(imgpath).convert('RGB')
     # print(img)
        img = self.data[index]
        if self.shape is not None:
            img = img.resize(self.shape)
        if self.transform is not None:
            img = self.transform(img)
        label = self.target[index]
        # print(label.type)
        label = torch.from_numpy(np.array(label, dtype=np.int64))

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
    # transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# test_data = []
# for i in range(len(test_image_path)):
#     img = Image.open(test_image_path[i]).convert('RGB')
#     test_data.append(img)

# test_loader = torch.utils.data.DataLoader(
#     listDataset_RAM(test_data, test_label, test_nSamples, shape=(init_width, init_height),
#                     shuffle=False,
#                     transform=transform_test,
#                     train=False,
#                     seen=0,
#                     batch_size=batch_size,
#                     num_workers=0),
#     batch_size=batch_size_test, shuffle=False, num_workers=8)

iteration = int(num_images/batch_size)
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
optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(epoch, coordinates, weights, bias, train_label_tensor):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    index_list = torch.randperm(num_images)

    for i in range(num_layers):
        weights[i] = weights[i][index_list]
        bias[i] = bias[i][index_list]
    weights[num_layers] = weights[num_layers][index_list]
    bias[num_layers] = bias[num_layers][index_list]
    train_label_tensor = train_label_tensor[index_list]
    for batch_idx in range(iteration):
        #inputs, targets = inputs.to(device), targets.to(device)
        st = time.time()
        for i in range(num_layers+1):
            if i == 0:
                output = coordinates.matmul(weights[i][batch_size * batch_idx: batch_size * (
                    batch_idx + 1)]) + bias[i][batch_size * batch_idx: batch_size * (batch_idx + 1)]
            else:
                output = output.matmul(weights[i][batch_size * batch_idx: batch_size * (
                    batch_idx + 1)]) + bias[i][batch_size * batch_idx: batch_size * (batch_idx + 1)]
            output = torch.sin(w0 * output)
        output = output[:, :, [2, 1, 0]]
        output = output.reshape(batch_size, image_w, image_h, 3)
        output = clamp_image(output)
        output = output.permute(0, 3, 1, 2)
        reconstructed_augment = transform_train(output)
        targets = train_label_tensor[batch_size *
                                     batch_idx: batch_size * (batch_idx + 1)]
        et = time.time()
        decode_aug_time.append(et-st)
        # ------------------------------------------------------------------------------------------------
        st = time.time()

        optimizer.zero_grad()
        outputs = net(reconstructed_augment)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        iter = iter + 1
        et = time.time()
        train_time.append(et-st)
    # print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
    #       (train_loss/(iter * 128), 100.*correct/total, correct, total))


# def test(epoch):
#     best_acc = 0
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     iter = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#             iter = iter + 1
#             # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             # % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
#           (test_loss/(iter * 100), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt_whole_pipeline.pth')
#         best_acc = acc

def gen_weights_bias_cord():
    linear_weights = []
    linear_bias = []
    linear_weights.append(torch.zeros(num_images, 2, layer_size))
    linear_bias.append(torch.zeros(num_images, layer_size))
    for i in range(num_layers-1):
        linear_weights.append(torch.zeros(num_images, layer_size, layer_size))
        linear_bias.append(torch.zeros(num_images, layer_size))
    linear_weights.append(torch.zeros(num_images, layer_size, 3))
    linear_bias.append(torch.zeros(num_images, 3))
    # load weight and bias
    for i in range(num_images):
        for j in range(num_layers):
            linear_weights[j][i] = weights[i]['net.%d.linear.weight' % j].t()
            linear_bias[j][i] = weights[i]['net.%d.linear.bias' % j].t()
        linear_weights[num_layers][i] = weights[i]['last_layer.linear.weight'].t()
        linear_bias[num_layers][i] = weights[i]['last_layer.linear.bias'].t()
    # change bia shape
    for i in range(num_layers+1):  # +1 becuase there's a last layer
        linear_bias[i] = linear_bias[i].view(-1, 1, linear_bias[i].shape[1])
    # move to device
    for i in range(num_layers+1):
        linear_bias[i] = linear_bias[i].to(device)
        linear_weights[i] = linear_weights[i].to(device)
    # gen_coordinates
    img = torch.zeros([batch_size, 3, image_w, image_h])
    coordinates = torch.zeros([batch_size, img.shape[2] * img.shape[3], 2])
    for i in range(batch_size):
        coordinates[i], _ = to_coordinates_and_features(img[i])
    coordinates = coordinates.to(device)

    return linear_weights, linear_bias, coordinates


if __name__ == "__main__":

    # get weights from disk to cpu:
    st = time.time()
    weights = []
    start_epoch = 0
    for i in range(num_images):
        INR_weights = torch.load(weight_dir + "best_model_%d.pt" %
                                 (i), map_location=torch.device('cpu'))
        weights.append(INR_weights)
    et = time.time()

    # get label list:
    train_label = np.load(label_path)
    # test_imgDir = './cifar_10_images/test_cifar10'
    # test_label = np.load("cifar_10_labels_test.npy")
    train_label_tensor = torch.from_numpy(train_label.astype(np.int64))
    # test_label_tensor = torch.from_numpy(test_label.astype(np.int64))

    train_label_tensor = train_label_tensor.to(device)
    # test_label_tensor = test_label_tensor.to(device)

    # get INR weights, bias, and corrd
    linear_weights, linear_bias, coordinates = gen_weights_bias_cord()
    disk_to_gpu_time.append(et-st)
    # ----------------------------------------------------------------------------------------
    for epoch in range(start_epoch, start_epoch+num_epoch):
        #lr = adjust_learning_rate(epoch)

        train(epoch, coordinates, linear_weights,
              linear_bias, train_label_tensor)
        # test(epoch)
        scheduler.step()
        print("The time of loading is:" + str(sum(disk_to_gpu_time)))
        print("The time of decoding+aug is:" + str(sum(decode_aug_time)))
        print("The time of training is:" + str(sum(train_time)))
        # print("time for forward is"+str(sum(forward_time)))
        # print("time for backward is"+str(sum(backward_time)))
