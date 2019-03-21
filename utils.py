import os
import struct
import numpy as np
from PIL import Image as image
import torch
import torchvision
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional as tf
from scipy.sparse import lil_matrix

def load_jpg(filename):
    return tf.to_tensor(image.open(filename)).unsqueeze(dim=0)


def unroll_matrix(X, m):
    flat_X = X.flatten()
    n = X.shape[0]
    unrolled_X = zeros(((n - m) ** 2, m**2))
    skipped = 0
    for i in range(n ** 2):
      if (i % n) < n - m and ((i / n) % n) < n - m:
          for j in range(m):
              for l in range(m):
                  unrolled_X[i - skipped, j * m + l] = flat_X[i + j * n + l]
      else:
          skipped += 1
    return unrolled_X

def unroll_kernel(kernel, n, sparse=False):
    m = kernel.shape[0]
    if sparse:
         unrolled_K = lil_matrix(((n - m)**2, n**2))
    else:
         unrolled_K = zeros(((n - m)**2, n**2))
    skipped = 0
    for i in range(n ** 2):
         if (i % n) < n - m and((i / n) % n) < n - m:
             for j in range(m):
                 for l in range(m):
                    unrolled_K[i - skipped, i + j * n + l] = kernel[j, l]
         else:
             skipped += 1
    return unrolled_K


def plot_activations(plt, tensor):
    plt.plot(numpy(tensor.flatten().sort()[0]))
    plt.savefig(tame_path.format("impala"))

def savefig(plt):
    plt.savefig(tame_path.format("impala"))

def numpy(tensor):
    return tensor.cpu().detach().numpy()

resize_transform = torchvision.transforms.Resize((224, 224))
def load_jpg(filename, resize=True):
    im = image.open(filename)
    if resize:
        im = resize_transform(im)
    im = tf.to_tensor(im).unsqueeze(dim=0).cuda()

    return im / im.max()

def random(size=224):
    return Variable(torch.rand(1, 3, size, size), requires_grad=True).cuda()

def vertical_gaussian_loss(x):
    loss = 0
    for i in range(223):
        loss += torch.norm(x[0, :, i, :] - x[0, :, i + 1, :])
        loss += torch.abs(x[0, :, i, :] - x[0, :, i + 1, :]).sum()

    return loss

def horizontal_gaussian_loss(x):
    loss = 0
    for i in range(223):
        loss += torch.norm(x[0, :, :, i] - x[0, :, :, i + 1])
        loss += torch.abs(x[0, :, :, i] - x[0, :, :, i + 1]).sum()

    return loss

def sharp_loss(x):
    n = x.size()[2]
    loss = 0
    for i in range(n - 1):
        loss += sharp(x[0, :, i, :] - x[0, :, i + 1, :]).sum()
    for i in range(n - 1):
        loss += sharp(x[0, :, :, i] - x[0, :, :, i + 1]).sum()

    return loss


def sharp(x, a=30, c=.5):
    return np.e ** (-a * (x - c) ** 2)

def tensor(x):
    return torch.FloatTensor(x)

def grid_gaussian_loss(x):
    loss = 0
    for i in range(0, 252, 4):
        for j in range(0, 252, 4):
            loss += torch.norm(x[0, :, i:i + 4, j:j + 4] - x[0, :, i + 1:i + 5, j + 1:j + 5])

    return loss

def conv_blur(x):
    kernel = 11
    stride = 4

    loss = 0
    for i in range(0, 256, stride):
        for j in range(0, 256, stride):
            loss += torch.var(x[0, :, i:i + kernel, j:j + kernel])

    return loss
