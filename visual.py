import numpy as np
import torch
import torchvision

class Visual():
    def __init__(self, model):
        self.model = model

    def __call__(self, x, layer):
        conv = self.model(x, end=layer)
        self.visual(conv)

    def visual(self, conv):
        conv = conv[0]
        spacing = 2

        n = int(np.ceil(np.sqrt(conv.size()[0])))
        size = int(conv.size()[1])

        image = torch.zeros(((size + 2 * spacing) * n, (size + 2 * spacing) * n))
        for i, lens in enumerate(conv):
            j = size * (i // n)
            k = (i % (n)) * size
            # print(i, j, k)
            image[k + spacing:k + size + spacing, j + spacing:j + size + spacing] = lens

        image = image.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 2, 2)
        image = torch.nn.functional.conv_transpose2d(image.clone(), kernel, stride=2, groups=1, padding=1)

        torchvision.utils.save_image(image, tame_path.format("tame"))

class FrontVisual():
    def __init__(self, model):
        self.model = model

    def __call__(self, x, layer):
        conv = self.model(x, end=layer)
        self.visual(conv)

    def visual(self, conv):
        conv = conv.sum(1)
        conv -= conv.min()
        conv /= conv.max()

        n = conv.size()[1]
        image = torch.rand(1, 3, n, n)
        image[0, 0, :, :] = conv
        image[0, 1, :, :] = conv
        image[0, 2, :, :] = conv

        torchvision.utils.save_image(image, tame_path.format("tame"))


def jpeg(tensor, side=0):
    scale = 3 * 224 // tensor.size()[2]
    kernel = torch.ones(3, 1, scale, scale)
    tensor = torch.nn.functional.conv_transpose2d(tensor.clone(), kernel, stride=scale, groups=3)

    filename = "impala" if side else "tame"
    torchvision.utils.save_image(tensor, tame_path.format(filename))
