import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class JigsawCrop(object):
    """
    The class implements the process of generating jigsaw crops for PIRL. The implementation is based on
    https://github.com/HobbitLong/PyContrast
    """

    def __init__(self, n_grid=2, img_size=512, crop_size=256):
        """
        Constructor, the function initializes the paramters.

        :param n_grid: Grid size to divide the original image
        :param img_size: Original image size
        :param crop_size: Jigsaw crop size
        """
        self.n_grid = n_grid
        self.img_size = img_size
        self.crop_size = crop_size
        self.grid_size = int(img_size / self.n_grid)
        self.side = self.grid_size - self.crop_size

        yy, xx = np.meshgrid(np.arange(n_grid), np.arange(n_grid))
        self.yy = np.reshape(yy * self.grid_size, (n_grid * n_grid,))
        self.xx = np.reshape(xx * self.grid_size, (n_grid * n_grid,))

    def __call__(self, img):
        """
        The function generates the jigsaw crops of a provided original image.

        :param img: Original image
        :return: Jigsaw crops
        """
        r_x = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        r_y = np.random.randint(0, self.side + 1, self.n_grid * self.n_grid)
        img = np.asarray(img, np.uint8)
        crops = []
        for i in range(self.n_grid * self.n_grid):
            crops.append(img[self.xx[i] + r_x[i]: self.xx[i] + r_x[i] + self.crop_size,
                         self.yy[i] + r_y[i]: self.yy[i] + r_y[i] + self.crop_size, :])
        crops = [Image.fromarray(crop) for crop in crops]
        return crops


class StackTransform(object):
    """
    The transform to group images independently.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        return torch.stack([self.transform(crop) for crop in imgs])


class JigsawTransform(object):
    """
    The implementation of generating jigsaw crops and torchvision transformation.
    """
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.Resize(1024),
             transforms.CenterCrop(512),
             transforms.RandomHorizontalFlip(),
             JigsawCrop(),
             StackTransform(transforms.Compose(
                 [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))]
        )

    def __call__(self, img):
        return [], self.transform(img)
