import torch
import time
from torchvision import models
from torch.autograd import Variable

# Check use GPU or not
use_gpu = torch.cuda.is_available()  # use GPU
print(use_gpu)
torch.manual_seed(123)
if use_gpu:
    torch.cuda.manual_seed(456)


# Define CNN Models:
model1 = models.resnet18(pretrained=True)
model2 = models.resnet18(pretrained=True)

# Eval Mode:
model1.eval()
model2.eval()

# Put on GPU:
if use_gpu:
    model1 = model1.cuda()
    model2 = model2.cuda()

# Create tmp Variable:
x = Variable(torch.randn(64, 3, 22, 224))
if use_gpu:
    x = x.cuda()


# Forward Pass:
tic1 = time.time()
out1 = model1(x)
#out2 = model2(x)
torch.cuda.synchronize()
tic2 = time.time()

sequential_forward_pass = tic2 - tic1
print('Time = ', sequential_forward_pass)  # example output --> Time =  0.6485