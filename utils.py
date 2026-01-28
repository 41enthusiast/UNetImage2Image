import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available.")
        print("GPU Device Name:", torch.cuda.get_device_name(0))  # Get the name of the first GPU device
        print("GPU Device Count:", torch.cuda.device_count())  # Get the number of available GPUs
        print("CUDA Version:", torch.version.cuda)  # Get the installed CUDA version
        print("PyTorch Version:", torch.__version__)  # Get the installed PyTorch version
    else:
        print("GPU is not available. Using CPU.")
        print(os.cpu_count(), 'CPU cores detected.')

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# def output_align(input, output):
#     """
#     In testing, sometimes output is several pixels less than irregular-size input,
#     here is to fill them.
#     """
#     if output.size() != input.size():
#         diff_width = input.size(-1) - output.size(-1)
#         diff_height = input.size(-2) - output.size(-2)
#         m = nn.ReplicationPad2d((0, diff_width, 0, diff_height))
#         output = m(output)

#     return output



def show_image_mask_grid(images, masks, nrow=4, img_name = 'imgmask_grid'):
    """
    images: (B, 3, H, W)
    masks:  (B, 1, H, W)
    """
    assert images.shape[0] == masks.shape[0]

    B, _, H, W = images.shape

    # Normalize masks to [0,1] for visualization
    masks_vis = masks.float()
    if masks_vis.max() > 1:
        masks_vis = (masks_vis / masks_vis.max()).clamp(min=1)

    # Convert masks to 3-channel so grid looks consistent
    masks_vis = masks_vis.repeat(1, 3, 1, 1)

    # Stack image above mask for each sample
    stacked = torch.cat([images, masks_vis], dim=2)
    # stacked shape: (B, 3, 2H, W)

    grid = vutils.make_grid(
        stacked,
        nrow=nrow,
        padding=2,
        normalize=True,
        # value_range=(-1, 1)  # remove if images already [0,1]
    )

    plt.figure(figsize=(12, 6))
    plt.imsave(f'logs/{img_name}.png',grid.permute(1, 2, 0),)

#got from edge connect repo
class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, device = 'cuda'):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19_reluvariant().to(device = device))
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        # G = torch.einsum('bij,bkj->bik', f, f) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        layer_names = ['relu2_2', 'relu3_4', 'relu4_4', 'relu5_2']
        for i in range(4):
            style_loss += self.criterion(self.compute_gram(x_vgg[layer_names[i]]), self.compute_gram(y_vgg[layer_names[i]]))
        # style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        # style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        # style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        # style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss_reluvariant(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], device = 'cuda'):
        super(PerceptualLoss_reluvariant, self).__init__()
        self.add_module('vgg', VGG19_reluvariant().to(device = device))
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(x_vgg[f'relu{i+1}_1'], y_vgg[f'relu{i+1}_1'])
        # content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        # content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        # content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        # content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

class PerceptualLoss_convvariant(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], device = 'cuda'):
        super(PerceptualLoss_convvariant, self).__init__()
        self.add_module('vgg', VGG19_convvariant().to(device = device))
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(x_vgg[f'conv{i+1}_1'], y_vgg[f'conv{i+1}_1'])
        # content_loss += self.weights[0] * self.criterion(x_vgg['conv1_1'], y_vgg['conv1_1'])
        # content_loss += self.weights[1] * self.criterion(x_vgg['conv2_1'], y_vgg['conv2_1'])
        # content_loss += self.weights[2] * self.criterion(x_vgg['conv3_1'], y_vgg['conv3_1'])
        # content_loss += self.weights[3] * self.criterion(x_vgg['conv4_1'], y_vgg['conv4_1'])
        # content_loss += self.weights[4] * self.criterion(x_vgg['conv5_1'], y_vgg['conv5_1'])

        return content_loss

class VGG19_reluvariant(torch.nn.Module):
    def __init__(self):
        super(VGG19_reluvariant, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            # 'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            # 'relu3_2': relu3_2,
            # 'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            # 'relu4_2': relu4_2,
            # 'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            # 'relu5_3': relu5_3,
            # 'relu5_4': relu5_4,
        }
        return out

class VGG19_convvariant(torch.nn.Module):
    def __init__(self):
        super(VGG19_convvariant, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()

        self.conv2_1 = torch.nn.Sequential()
        self.conv2_2 = torch.nn.Sequential()

        self.conv3_1 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv3_3 = torch.nn.Sequential()
        self.conv3_4 = torch.nn.Sequential()

        self.conv4_1 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()
        self.conv4_3 = torch.nn.Sequential()
        self.conv4_4 = torch.nn.Sequential()

        self.conv5_1 = torch.nn.Sequential()
        self.conv5_2 = torch.nn.Sequential()
        self.conv5_3 = torch.nn.Sequential()
        self.conv5_4 = torch.nn.Sequential()

        for x in range(1):
            self.conv1_1.add_module(str(x), features[x])

        for x in range(1, 3):
            self.conv1_2.add_module(str(x), features[x])

        for x in range(3, 6):
            self.conv2_1.add_module(str(x), features[x])

        for x in range(6, 8):
            self.conv2_2.add_module(str(x), features[x])

        for x in range(8, 11):
            self.conv3_1.add_module(str(x), features[x])

        for x in range(11, 13):
            self.conv3_2.add_module(str(x), features[x])

        for x in range(13, 15):
            self.conv3_3.add_module(str(x), features[x])

        for x in range(15, 17):
            self.conv3_4.add_module(str(x), features[x])

        for x in range(17, 20):
            self.conv4_1.add_module(str(x), features[x])

        for x in range(20, 22):
            self.conv4_2.add_module(str(x), features[x])

        for x in range(22, 24):
            self.conv4_3.add_module(str(x), features[x])

        for x in range(24, 26):
            self.conv4_4.add_module(str(x), features[x])

        for x in range(26, 29):
            self.conv5_1.add_module(str(x), features[x])

        for x in range(29, 31):
            self.conv5_2.add_module(str(x), features[x])

        for x in range(31, 33):
            self.conv5_3.add_module(str(x), features[x])

        for x in range(33, 35):
            self.conv5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)

        conv2_1 = self.conv2_1(conv1_2)
        conv2_2 = self.conv2_2(conv2_1)

        conv3_1 = self.conv3_1(conv2_2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_4 = self.conv3_4(conv3_3)

        conv4_1 = self.conv4_1(conv3_4)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_4 = self.conv4_4(conv4_3)

        conv5_1 = self.conv5_1(conv4_4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_4 = self.conv5_4(conv5_3)

        out = {
            'conv1_1': conv1_1,
            # 'conv1_2': conv1_2,

            'conv2_1': conv2_1,
            # 'conv2_2': conv2_2,
            
            'conv3_1': conv3_1,
            # 'conv3_2': conv3_2,
            # 'conv3_3': conv3_3,
            # 'conv3_4': conv3_4,
            
            'conv4_1': conv4_1,
            # 'conv4_2': conv4_2,
            # 'conv4_3': conv4_3,
            # 'conv4_4': conv4_4,

            'conv5_1': conv5_1,
            # 'conv5_2': conv5_2,
            # 'conv5_3': conv5_3,
            # 'conv5_4': conv5_4,
        }
        return out

if __name__ == '__main__':
    import time

    check_gpu()
    set_seed(42)
    s_loss = StyleLoss()
    p_loss = PerceptualLoss_convvariant(device='cuda')
    x = torch.randn((1,1,224,224)).to('cuda')
    y = torch.randn((1,1,224,224)).to('cuda')

    start = time.time()
    for i in range(10):
        _ = s_loss(x,y)
        _ = p_loss(x,y)
    end = time.time()
    print('Average Time for an iteration:', (end - start)/10, 'seconds')

    print('Style Loss:', s_loss(x,y).item())
    print('Perceptual Loss:', p_loss(x,y).item())