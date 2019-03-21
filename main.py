import os
from time import time
import numpy as np
from numpy.linalg import pinv, matrix_rank, norm
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from PIL import Image as image
import torch
import torchvision
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.functional import mse_loss, softmax
import torch.utils.model_zoo as model_zoo
import torchvision.transforms.functional as tf

from labels import labels
from utils import *
from alexnet import *
from solution_space import *
from visual import *

class Generator():
    def __init__(self, model):
        self.model = model
        self.shapes = [model(torch.rand(1, 3, 224, 224), end=i).size() for i in range(23)]

        self.x_min = torch.zeros(1, 3, 224, 224).cuda()
        self.x_max = torch.ones(1, 3, 224, 224).cuda()

        self.start = 0
        self.end = 23

    def __call__(self, x, h=.1, steps=1):
        return self.backward_generate(target, h=h, steps=steps)

    def quick_generate(self, target):
        y = Variable(torch.rand(self.shapes[1]), requires_grad=True)
        loss = lambda x: self.loss(x, y=target, start=1)
        constraint = lambda x: self.constraint(x, mini=0, maxi=5)
        y = grad_ascent(y, loss, constraint=constraint, h=1, steps=100, verbose=False)

        start = time()
        total_loss = 0

        n_runs = 100
        for _ in range(n_runs):
            x = Variable(torch.rand(self.shapes[0]), requires_grad=True)
            loss = lambda x: self.loss(x, y=y, end=1)
            constraint = lambda x: self.constraint(x)
            x = grad_ascent(x, loss, constraint=constraint, h=.1, steps=100, verbose=False)
            jpeg(x)

            total_loss += self.loss(x, y=target).item()

        print(total_loss / n_runs, time() - start)

        start = time()
        total_loss = 0

        for _ in range(n_runs):
            x = Variable(torch.rand(self.shapes[0]), requires_grad=True)
            loss = lambda x: self.loss(x, y=target)
            constraint = lambda x: self.constraint(x)
            x = grad_ascent(x, loss, constraint=constraint, h=.1, steps=1000, verbose=False)

            total_loss += self.loss(x, y=target).item()

        print(total_loss / n_runs, time() - start)

        return y


    def backward_generate(self, target, inter_layers=[], h=1, steps=1000):
        j = self.end
        y_j = target

        for i in reversed(inter_layers):
            y_i = Variable(torch.rand(self.shapes[i]), requires_grad=True)
            loss = lambda x: self.loss(x, y=y_j, start=i, end=j)
            constraint = lambda x: self.constraint(x, mini=-20, maxi=20)
            y_i = grad_ascent(y_i, loss, constraint=constraint, h=h, steps=steps, verbose=False)

            y_j = y_i.data.clone()
            j = i

        x = Variable(torch.rand(self.shapes[0]), requires_grad=True)
        loss = lambda x: self.loss(x, y=y_j, start=0, end=j)
        x = grad_ascent(x, loss, constraint=self.constraint, h=h, steps=steps, verbose=False)

        return x

    def loss(self, x, y=None, start=0, end=23):
        loss = 0
        loss += mse_loss(self.model(x, start=start, end=end), y)

        return loss

    def constraint(self, x, mini=0, maxi=1):
        x_min = mini * torch.ones(x.size())
        x_max = maxi * torch.ones(x.size())

        x = torch.max(x, x_min)
        x = torch.min(x, x_max)

        return x


def grad_ascent(x, loss, constraint=None, h=.01, steps=1, verbose=True, model_loss=None, i=0):
    if verbose:
        print("\n\tGRAD ASCENT")
        print("================================================================")

    x_start = x.clone()
    for i in range(steps):
        step_loss = loss(x)
        if verbose:
            print("loss:\t\t{}".format(step_loss.data.item()))

        step_loss.backward(retain_graph=True)

        grad, *_ = x.grad.data
        x_step = x - h * grad / grad.norm()
        if constraint:
            x_step = constraint(x_step)


        if verbose:
            print("grad mag:\t{}".format(grad.norm().item()))
            print("step mag:\t{}".format(h * grad.norm().item()))
            print()

        x = Variable(x_step.data, requires_grad=True)

    if verbose:
        print("final loss:", loss(x).data.item())
        print("displacement:", torch.norm(x_start - x).item())
        print("========================================================\n")

    return x

def load_synset(files, n=-1):
    synset = []
    for file in files[:n]:
        synset.append(load_jpg("synset/" + file))
    return synset

def perfect_target(i):
    t = torch.zeros(1, 1000)
    t[0][i] = 1

    return t

# environment setup ===============================================
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# alexnet model ===============================================
model = alexnet().to(device)
model.cuda()
model.eval()

# run =========================================================
gen = Generator(model)

# Generative Accuracy Experiment
targets = [
    perfect_target(54),
    perfect_target(120),
    perfect_target(255),
    perfect_target(386),
    perfect_target(954),
]
layer_sets = [
    [],
    [3],
    [10],
    [3, 6],
    [6, 14],
    [3, 6, 10],
    [3, 10, 14],
    [3, 6, 10, 14]
]
n_iters = 100
with open("results.txt", 'w') as f:
    for i, target in enumerate(targets):
        for j, layer_set in enumerate(layer_sets):
            total_loss = 0
            start = time()

            for _ in range(n_iters):
                x = gen.backward_generate(target, inter_layers=layer_set)
                loss = gen.loss(x, y=target).item()
                total_loss += loss

            print("target: {}\t layer_set: {}\t loss: {}\t time: {}"
                .format(i, j, total_loss / n_iters, time() - start))
        print()

# Generative Speed Experiment
y = gen.quick_generate(t)
