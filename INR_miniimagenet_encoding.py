import numpy as np
import cv2

import time
import torch

from torch import nn
from math import sqrt

from typing import Dict

from torch._C import dtype

from torchvision import transforms

from collections import OrderedDict

#import tqdm

from typing import Any, BinaryIO, List, Optional, Tuple, Union

import pathlib

from types import FunctionType

from PIL import Image, ImageColor, ImageDraw, ImageFont

import os
import argparse

import random

from statistics import stdev
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("-dp", "--dataset_path", help="The path of the dataset", default=f"/export/hdd/scratch/dataset/mini_imagenet_split/train/")
parser.add_argument("-ld", "--logdir", help="Path to save logs", default=f"./mini_ImageNet_logfiles_40_final")
parser.add_argument("-wd", "--INR_weights_dir", help="Path to save INR weights", default=f"./weights_mini_ImageNet_40_final/trail")
parser.add_argument("-ri", "--resized_image", help="Path to save resized images", default=f"./resizeimg_mini_imagenet_40_final")
parser.add_argument("-rconi", "--reconstructed_image", help="Path to save reconstructed images", default=f"./reconimages_mini_ImageNet_40_final/trial")
parser.add_argument("-ra", "--reconstructed_image_array", help="Path to save reconstructed images array", default=f"./mini_ImageNet_array_40_final")
parser.add_argument("-imgw", "--image_w", help="The width of reconstructed image", type = int, default= 224)
parser.add_argument("-imgh", "--image_h", help="The height of reconstructed image", type = int, default= 224)
parser.add_argument("-device", "--device", help="The id of the GPU", default=f"cuda:0")

parser.add_argument("-nit", "--num_iters", help="Number of iterations to train for", type=int, default=5000)
parser.add_argument("-nim", "--num_images", help="Number of images to train on", type=int, default=7000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=5e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
#parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
#parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=40)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)
parser.add_argument("-bs", "--bs_number", help="the index of the image to training.", type=int, default=0)




args = parser.parse_args()


layer_size = args.layer_size
num_layers = args.num_layers

image_w = args.image_w
image_h = args.image_h
w0 = args.w0
w0_initial = args.w0_initial

learning_rate = args.learning_rate
num_iters = args.num_iters
device = args.device

start_idx = args.bs_number


#train_set = '/export/hdd/scratch/dataset/imagenet/train/n01440764' ##
train_set = args.dataset_path

#logdir = './weights_ImageNet'
logdir = args.INR_weights_dir
log_dir_resize = args.resized_image
log_dir_image = args.reconstructed_image
logdir_array = args.reconstructed_image_array
logdir_logfiles = args.logdir

if not os.path.exists(logdir):
    os.makedirs(logdir)
# ** Insert line to create /resizeimg_in if it doesn't exist
#log_dir_resize = './resizeimg_in'
if not os.path.exists(log_dir_resize):
    os.makedirs(log_dir_resize)
#log_dir_image = './reconimages_ImageNet/trial'
if not os.path.exists(log_dir_image):
    os.makedirs(log_dir_image)
#logdir_array = './ImageNet_array'
if not os.path.exists(logdir_array):
    os.makedirs(logdir_array)
#logdir_logfiles = './ImageNet_logfiles'
if not os.path.exists(logdir_logfiles):
    os.makedirs(logdir_logfiles)
    

    
    
DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}    
    
    
    
def is_image_file(filename): # Compares 'filename' extension to common image file types.
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def load_image_path(imgDir):

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
    print(train_imageNames[0])
    print(train_files[0])
    
    return train_files, train_imageNames

def save_image_new(image, address):
    image = image * 255
    image = image.round()
    image = torch.clamp(image,0,255)
    image_array = image.cpu().numpy()
    image_array = image_array.transpose(1,2,0)
    cv2.imwrite(address, image_array) 


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.
    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))

def bpp(image, model):
    """Computes size in bits per pixel of model.
    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels

def psnr(img1, img2):
    """Calculates PSNR between two images.
    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()

def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.
    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.

def get_clamped_psnr(img_recon, img):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.
    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(clamp_image(img_recon), img)

@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

    
def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")
    
    
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(make_grid)
    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


train_files, train_imageNames = load_image_path(train_set)
train_image_num = len(train_files)
train_image_num = args.num_images

#image_w = 224
#image_h = 224

image = np.zeros([train_image_num,3,image_w,image_h])
#image_test = np.zeros([10000,3,32,32])
idx_offset = start_idx
#time_start = time.time()
for i in range (train_image_num):
    
    img = cv2.imread(train_files[idx_offset + i])
    
    dim = (image_w, image_h)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = np.transpose(img, (2, 0, 1))
    image[i] = img
    
dir_appendix = f'_{num_layers}x{layer_size}_{w0:.0f}fq_{num_iters}ep_{train_image_num}im'
logfile = open(logdir_logfiles + '/logfile' + dir_appendix + '.txt','w')
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': []}

train_times = []
avg_psnr = 0
total_train_time = 0
avg_train_time = 0
model_size = 0
logfile.write('image_num total_inr_fitting_time best_training_psnr\n')


class Distribution():
    def __init__(self, init=0):
        super().__init__()
        self.vals = []
        self.stats = {'max': 0, 'min': 0, 'mean': 0, 'stdev': 0}
    def get_stats(self):
        self.stats['max'] = max(self.vals)
        self.stats['min'] = min(self.vals)
        self.stats['mean'] = mean(self.vals)
        self.stats['stdev'] = stdev(self.vals)





class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    """Implements a single SIREN layer.
    Args:
        dim_in (int): Dimension of input.
        dim_out (int): Dimension of output.
        w0 (float):
        c (float): c value from SIREN paper used for weight initialization.
        is_first (bool): Whether this is first layer of model.
        use_bias (bool):
        activation (torch.nn.Module): Activation function. If None, defaults to
            Sine activation.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
                 use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # Initialize layers following SIREN paper
        w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class Siren(nn.Module):
    """SIREN model.
    Args:
        dim_in (int): Dimension of input.
        dim_hidden (int): Dimension of hidden layers.
        dim_out (int): Dimension of output.
        num_layers (int): Number of layers.
        w0 (float): Omega 0 from SIREN paper.
        w0_initial (float): Omega 0 for first layer.
        use_bias (bool):
        final_activation (torch.nn.Module): Activation function.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30.,
                 w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SirenLayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0,
                                use_bias=use_bias, activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)
    
    
class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.
        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

    def train(self, coordinates, features, num_iters):
        """Fit neural net to image.
        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        #with tqdm.trange(num_iters, ncols=100) as t:
        for i in range (num_iters):
            # Update model
            self.optimizer.zero_grad()
            predicted = self.representation(coordinates)
            loss = self.loss_func(predicted, features)
            loss.backward()
            self.optimizer.step()

            # Calculate psnr
            psnr = get_clamped_psnr(predicted, features)

            # Print results and update logs
            log_dict = {'loss': loss.item(),
                        'psnr': psnr,
                        'best_psnr': self.best_vals['psnr']}
            #t.set_postfix(**log_dict)
            for key in ['loss', 'psnr']:
                self.logs[key].append(log_dict[key])

            # Update best values
            if loss.item() < self.best_vals['loss']:
                self.best_vals['loss'] = loss.item()
            if psnr > self.best_vals['psnr']:
                self.best_vals['psnr'] = psnr
                # If model achieves best PSNR seen during training, update
                # model
                if i > int(num_iters / 2.):
                    for k, v in self.representation.state_dict().items():
                        self.best_model[k].copy_(v)
            self.scheduler.step()

psnr_distr = Distribution(0.0)
loss_distr = Distribution(1e8)
train_time_distr = Distribution()


for i in range (train_image_num):
    print("the image is:",i + start_idx)
    logfile.write(f'{i + start_idx} ')
    
    func_rep = Siren(
            dim_in=2,
            dim_hidden=layer_size,
            dim_out=3,
            num_layers=num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=w0_initial,
            w0=w0
        ).to(device)
    dtype = torch.float32
    img = torch.from_numpy(image[i]/255.0).to(device, dtype)
    coordinates, features = to_coordinates_and_features(img)
    coordinates, features = coordinates.to(device), features.to(device)
    model_size = model_size_in_bits(func_rep) / 8192.
    #print(f'Model size: {model_size:.1f}kB')
    # Dictionary to register mean values (both full precision and half precision)
    
    #fp_bpp = bpp(model=func_rep, image=img)
    #print(fp_bpp)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    trainer = Trainer(func_rep, lr=learning_rate)
    trainer.train(coordinates, features, num_iters=num_iters)
    end.record()
    torch.cuda.synchronize()
    train_time = start.elapsed_time(end)
    train_times.append(train_time)
    total_train_time += train_times[i]
    
    logfile.write(f'{train_time:.18f} ')
    

    
    logfile.write(f'{trainer.best_vals["psnr"]:.2f}\n')
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    
    if i == train_image_num - 1:
        #print(func_rep)
        logfile.write(f'\n{func_rep}\n\n')
        fp_bpp = bpp(model=func_rep, image=img)
        results['fp_bpp'].append(fp_bpp)
    
    if not os.path.exists(logdir + dir_appendix):
        os.makedirs(logdir + dir_appendix)
    
    results['fp_psnr'].append(trainer.best_vals['psnr'])
    avg_psnr += results['fp_psnr'][i]
    torch.save(trainer.best_model, logdir + dir_appendix + f'/best_model_{start_idx + i}.pt')
    func_rep.load_state_dict(trainer.best_model)
    
    with torch.no_grad():
        img_recon = func_rep(coordinates).reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        #save_image(torch.clamp(img_recon, 0, 1).to('cpu'), log_dir_image + f'/fp_reconstruction_{i}.png')
        img_array = img_recon.cpu().numpy()
        np.save(logdir_array + f'/fp_reconstruction_{start_idx + i}.npy', img_array)
        
        if not os.path.exists(log_dir_image + dir_appendix):
            os.makedirs(log_dir_image + dir_appendix)
        save_image_new(img_recon, log_dir_image + dir_appendix + f'/fp_reconstruction_{start_idx + i}.png')
    #print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    psnr_distr.vals.append(trainer.best_vals['psnr'])
    loss_distr.vals.append(trainer.best_vals['loss'])
    train_time_distr.vals.append(train_time)
psnr_distr.get_stats()
loss_distr.get_stats()
train_time_distr.get_stats()
    
avg_psnr = avg_psnr / train_image_num
avg_train_time = total_train_time / train_image_num
logfile.write(f'Summary:\n')
logfile.write(f'Num layers: {num_layers}\n')
logfile.write(f'Layer size: {layer_size}\n')
logfile.write(f'Num iters: {num_iters}\n')
logfile.write(f'Size of dataset: {train_image_num}\n\n')
logfile.write(f'Compressed model size: {model_size:.1f}kB\n')
bpp_out = results['fp_bpp'][0]
logfile.write(f'Bits per pixel(fp_bpp): {bpp_out}\n\n')
for key in ['mean', 'stdev', 'max', 'min']:
    datum = psnr_distr.stats[key]
    logfile.write(f'PSNR {key}: {datum}\n')
logfile.write('\n')
for key in ['mean', 'stdev', 'max', 'min']:
    datum = loss_distr.stats[key]
    logfile.write(f'Loss {key}: {datum}\n')
logfile.write('\n')
for key in ['mean', 'stdev', 'max', 'min']:
    datum = train_time_distr.stats[key]
    logfile.write(f'Train time {key}: {datum}ms\n')
logfile.write(f' Total training time: {total_train_time}ms\n')
logfile.close()