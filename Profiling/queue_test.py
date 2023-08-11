import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from torchvision import models
import numpy as np
import time
from PIL import Image
import torch.utils.data as Data
import re
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from queue import Queue

num_images = 64
num_inter = 20019
hidden_dimension = 20
weights = []
batch_size = 64
start_epoch = 0
lr = 0.01
test_nSamples = 10000
init_width = 224
init_height = 224
batch_size_test = 100
device = 'cuda:1'

w_dir = "/nethome/hyang628/projects/inr/INR/dataMoveProfile/weights_1/"

test_imgDir = './cifar_10_images/test_cifar10'

disk_to_gpu_time = []
decode_aug_time = []
train_time = []


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def load_image_path_test(imgDir):

    all_training_files = os.walk(imgDir)
    train_files = []
    train_imageNames = []
    train_nSamples = 0
    for path, direction, filelist in all_training_files:
        files = [
            file for file in filelist
            if os.path.isfile(os.path.join(path, file))
        ]
        imageNames = [
            file.split('.')[0] for file in files if is_image_file(file)
        ]
        files = [
            os.path.join(path, file) for file in files if is_image_file(file)
        ]
        train_files.append(files)
        train_imageNames.append(imageNames)
        train_nSamples = train_nSamples + len(files)
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
    def __init__(self,
                 data,
                 target,
                 nsamples,
                 shape=None,
                 shuffle=True,
                 transform=None,
                 target_transform=None,
                 train=False,
                 seen=0,
                 batch_size=224,
                 num_workers=0):

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
    return 20. * np.log10(1.) - 10. * (
        img1 - img2).pow(2).mean().log10().to('cpu').item()


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


def decode(batch_num, coordinates, linear_0_weight, linear_0_bias,
           linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias,
           linear_3_weight, linear_3_bias, linear_4_weight, linear_4_bias,
           buf):

    for batch_idx in range(batch_num):
        output1 = coordinates.matmul(
            linear_0_weight[batch_size * batch_idx:batch_size *
                            (batch_idx + 1)]
        ) + linear_0_bias[batch_size * batch_idx:batch_size * (batch_idx + 1)]
        output1 = torch.sin(30 * output1)
        output2 = output1.matmul(
            linear_1_weight[batch_size * batch_idx:batch_size *
                            (batch_idx + 1)]
        ) + linear_1_bias[batch_size * batch_idx:batch_size * (batch_idx + 1)]
        output2 = torch.sin(30 * output2)

        output3 = output2.matmul(
            linear_2_weight[batch_size * batch_idx:batch_size *
                            (batch_idx + 1)]
        ) + linear_2_bias[batch_size * batch_idx:batch_size * (batch_idx + 1)]
        output3 = torch.sin(30 * output3)

        output4 = output3.matmul(
            linear_3_weight[batch_size * batch_idx:batch_size *
                            (batch_idx + 1)]
        ) + linear_3_bias[batch_size * batch_idx:batch_size * (batch_idx + 1)]
        output4 = torch.sin(30 * output4)

        output5 = output4.matmul(
            linear_4_weight[batch_size * batch_idx:batch_size *
                            (batch_idx + 1)]
        ) + linear_4_bias[batch_size * batch_idx:batch_size * (batch_idx + 1)]
        # print(output4.shape)
        output5 = output5[:, :, [2, 1, 0]]
        output5 = output5.reshape(batch_size, 224, 224, 3)
        output5 = clamp_image(output5)
        output = output5.permute(0, 3, 1, 2)

        buf.put(output)
    return output


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
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_label = torch.ones(num_images * num_inter).type(torch.int64)
train_label_tensor = train_label.to(device)

# test_data = []
# for i in range(len(test_image_path)):
#     img = Image.open(test_image_path[i]).convert('RGB')
#     test_data.append(img)

# test_loader = torch.utils.data.DataLoader(listDataset_RAM(
#     test_data,
#     test_label,
#     test_nSamples,
#     shape=(init_width, init_height),
#     shuffle=False,
#     transform=transform_test,
#     train=False,
#     seen=0,
#     batch_size=batch_size,
#     num_workers=0),
#                                           batch_size=batch_size_test,
#                                           shuffle=False,
#                                           num_workers=8)

# net = VGG('VGG19')
net = models.resnet18()
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


def train(epoch, coordinates, linear_0_weight, linear_0_bias, linear_1_weight,
          linear_1_bias, linear_2_weight, linear_2_bias, linear_3_weight,
          linear_3_bias, linear_4_weight, linear_4_bias, train_label_tensor):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iter = 0
    index_list = torch.randperm(num_images * num_inter)
    linear_0_weight = linear_0_weight[index_list]
    linear_0_bias = linear_0_bias[index_list]
    linear_1_weight = linear_1_weight[index_list]
    linear_1_bias = linear_1_bias[index_list]
    linear_2_weight = linear_2_weight[index_list]
    linear_2_bias = linear_2_bias[index_list]
    linear_3_weight = linear_3_weight[index_list]
    linear_3_bias = linear_3_bias[index_list]
    linear_4_weight = linear_4_weight[index_list]
    linear_4_bias = linear_4_bias[index_list]
    train_label_tensor = train_label_tensor[index_list]

    batch_buf = mp.Queue(maxsize=10
                         )
    p = mp.Process(target=decode,
                   args=(20019, coordinates, linear_0_weight, linear_0_bias,
                         linear_1_weight, linear_1_bias, linear_2_weight,
                         linear_2_bias, linear_3_weight, linear_3_bias,
                         linear_4_weight, linear_4_bias, batch_buf))
    p.start()
    for batch_idx in range(num_inter):
        #inputs, targets = inputs.to(device), targets.to(device)
        start_time = time.time()

        # --------------------------------decode + aug-----------------------------------------------
        ts_dec = time.time()
        output = batch_buf.get()
        reconstructed_augment = transform_train(output)
        te_dec = time.time()
        decode_aug_time.append(te_dec - ts_dec)

        # ------------------------------------ train ---------------------------------------------------
        ts_train = time.time()
        targets = train_label_tensor[batch_size * batch_idx:batch_size *
                                     (batch_idx + 1)]
        optimizer.zero_grad()
        outputs = net(reconstructed_augment)
        ##
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        iter = iter + 1
        te_train = time.time()
        train_time.append(te_train - ts_train)
        # p.join()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        # % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    p.join()
    print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
          (train_loss / (iter * 128), 100. * correct / total, correct, total))


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
#             #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#     print('Loss: %.8f | Acc: %.8f%% (%d/%d)' %
#           (test_loss / (iter * 100), 100. * correct / total, correct, total))

#     # Save checkpoint.
#     acc = 100. * correct / total
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

if __name__ == "__main__":
    print("---------------start--------------------")
    mp.set_start_method("forkserver")
    # ---------------------------------------load weights---------------------------------------------------------------------------
    ts_load = time.time()
    for i in os.listdir(w_dir):
        INR_weights = torch.load("%s%s" % (w_dir, i),
                                 map_location=torch.device('cpu'))
        weights.append(INR_weights)
    linear_0_weight = torch.zeros(num_images * num_inter, 2,
                                  hidden_dimension).share_memory_()
    linear_0_bias = torch.zeros(num_images * num_inter,
                                hidden_dimension).share_memory_()
    linear_1_weight = torch.zeros(num_images * num_inter, hidden_dimension,
                                  hidden_dimension).share_memory_()
    linear_1_bias = torch.zeros(num_images * num_inter,
                                hidden_dimension).share_memory_()
    linear_2_weight = torch.zeros(num_images * num_inter, hidden_dimension,
                                  hidden_dimension).share_memory_()
    linear_2_bias = torch.zeros(num_images * num_inter,
                                hidden_dimension).share_memory_()
    linear_3_weight = torch.zeros(num_images * num_inter, hidden_dimension,
                                  hidden_dimension).share_memory_()
    linear_3_bias = torch.zeros(num_images * num_inter, hidden_dimension)
    linear_4_weight = torch.zeros(num_images * num_inter, hidden_dimension,
                                  3).share_memory_()
    linear_4_bias = torch.zeros(num_images * num_inter, 3).share_memory_()

    # shared queue

    for i in range(num_images * num_inter):
        r_i = i % num_images
        linear_0_weight[i] = weights[r_i]['net.0.linear.weight'].t()
        linear_0_bias[i] = weights[r_i]['net.0.linear.bias']
        linear_1_weight[i] = weights[r_i]['net.1.linear.weight'].t()
        linear_1_bias[i] = weights[r_i]['net.1.linear.bias']
        linear_2_weight[i] = weights[r_i]['net.2.linear.weight'].t()
        linear_2_bias[i] = weights[r_i]['net.2.linear.bias']
        linear_3_weight[i] = weights[r_i]['net.3.linear.weight'].t()
        linear_3_bias[i] = weights[r_i]['net.3.linear.bias']
        linear_4_weight[i] = weights[r_i]['last_layer.linear.weight'].t()
        linear_4_bias[i] = weights[r_i]['last_layer.linear.bias']

    img = torch.zeros([batch_size, 3, 224, 224])
    coordinates = torch.zeros([batch_size, img.shape[2] * img.shape[3], 2])
    for i in range(batch_size):
        coordinates[i], _ = to_coordinates_and_features(img[i])

    # coordinates = coordinates.share_memory_()
    coordinates = coordinates.to(device)

    linear_0_bias = linear_0_bias.view(-1, 1, linear_0_bias.shape[1])
    linear_1_bias = linear_1_bias.view(-1, 1, linear_1_bias.shape[1])
    linear_2_bias = linear_2_bias.view(-1, 1, linear_2_bias.shape[1])
    linear_3_bias = linear_3_bias.view(-1, 1, linear_3_bias.shape[1])
    linear_4_bias = linear_4_bias.view(-1, 1, linear_4_bias.shape[1])

    linear_0_weight = linear_0_weight.to(device)
    linear_0_bias = linear_0_bias.to(device)
    linear_1_weight = linear_1_weight.to(device)
    linear_1_bias = linear_1_bias.to(device)
    linear_2_weight = linear_2_weight.to(device)
    linear_2_bias = linear_2_bias.to(device)
    linear_3_weight = linear_3_weight.to(device)
    linear_3_bias = linear_3_bias.to(device)
    linear_4_weight = linear_4_weight.to(device)
    linear_4_bias = linear_4_bias.to(device)

    te_load = time.time()
    disk_to_gpu_time.append(te_load - ts_load)
    # ------------------------------------------------------------------------------------------------

    for epoch in range(start_epoch, start_epoch + 35):
        #lr = adjust_learning_rate(epoch)
        stime = time.time()
        train(epoch, coordinates, linear_0_weight, linear_0_bias,
              linear_1_weight, linear_1_bias, linear_2_weight, linear_2_bias,
              linear_3_weight, linear_3_bias, linear_4_weight, linear_4_bias,
              train_label_tensor)
        print("The" + str(epoch) + " epoch spent time:" +
              str(time.time() - stime))
        # test(epoch)

        scheduler.step()
        print("The total time of loading is:" + str(sum(disk_to_gpu_time)))
        print("The total time of decoding+aug is:" + str(sum(decode_aug_time)))
        print("The total time of training is:" + str(sum(train_time)))
