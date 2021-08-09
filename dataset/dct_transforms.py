import cv2
import numpy as np
import jpeg2dct.numpy
from turbojpeg import TurboJPEG
from configs.dct_config import *
import torch
from configs import dct_subsets
import random
import math
from dataset import dct_functional as F
import collections

# return the origin image and the upscaled image.
# As CbCr is smaller than Y, but we want the results of DCT are equal-shaped.
class DCT_Upscale(object):
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        return img, F.cv2_upscale(img, self.upscale_factor, self.interpolation)

# transform to DCT - TransformUpscaledDCT
class TransformToDCT(object):
    def __init__(self):
        self.jpeg_encoder = TurboJPEG(libjpeg_path)

    def __call__(self, img):
        y, cbcr = img[0], img[1]
        dct_y, _, _ = F.transform_to_dct(y, self.jpeg_encoder)
        _, dct_cb, dct_cr = F.transform_to_dct(
            cbcr, self.jpeg_encoder)  # do DCT saperately for Y and CbCr, as they are in different shapes.
        return dct_y, dct_cb, dct_cr

# To Tensor
class ToTensorDCT(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        y, cb, cr = img[0], img[1], img[2]
        y, cb, cr = F.to_tensor_dct(y), F.to_tensor_dct(cb), F.to_tensor_dct(cr)

        return y, cb, cr


class SubsetDCT(object):
    def __init__(self, channels=20, pattern='square'):
        self.channels = channels

        if pattern == 'square':
            self.subset_channel_index = dct_subsets.subset_channel_index_square
        elif pattern == 'learned':
            self.subset_channel_index = dct_subsets.subset_channel_index_learned
        elif pattern == 'triangle':
            self.subset_channel_index = dct_subsets.subset_channel_index_triangle

        if self.channels < 192:
            self.subset_y = self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cr = self.subset_channel_index[channels][2]

    def __call__(self, tensor):
        if self.channels < 192:
            dct_y, dct_cb, dct_cr = tensor[0], tensor[1], tensor[2]
            dct_y, dct_cb, dct_cr = dct_y[self.subset_y], dct_cb[self.subset_cb], dct_cr[self.subset_cr]
            return dct_y, dct_cb, dct_cr
        else:
            return tensor[0], tensor[1], tensor[2]


class Aggregate(object):
    def __call__(self, img):
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        value = torch.cat((dct_y, dct_cb, dct_cr), dim=0)
        return value


class NormalizeDCT(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, y_mean, y_std, cb_mean=None, cb_std=None, cr_mean=None, cr_std=None, channels=None, pattern='square'):
        self.y_mean,  self.y_std = y_mean, y_std
        self.cb_mean, self.cb_std = cb_mean, cb_std
        self.cr_mean, self.cr_std = cr_mean, cr_std

        if channels < 192:
            if pattern == 'square':
                self.subset_channel_index = dct_subsets.subset_channel_index_square
            elif pattern == 'learned':
                self.subset_channel_index = dct_subsets.subset_channel_index_learned
            elif pattern == 'triangle':
                self.subset_channel_index = dct_subsets.subset_channel_index_triangle

            self.subset_y = self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cb = [64+c for c in self.subset_cb]
            self.subset_cr = self.subset_channel_index[channels][2]
            self.subset_cr = [128+c for c in self.subset_cr]
            self.subset = self.subset_y + self.subset_cb + self.subset_cr
            self.mean_y, self.std_y = [y_mean[i] for i in self.subset], [
                y_std[i] for i in self.subset]
        else:
            self.mean_y, self.std_y = y_mean, y_std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y = F.normalize(y,  self.y_mean,  self.y_std)
            cb = F.normalize(cb, self.cb_mean, self.cb_std)
            cr = F.normalize(cr, self.cr_mean, self.cr_std)
            return y, cb, cr
        else:
            y = F.normalize(tensor, self.mean_y, self.std_y)
            return y, None, None


# accept numpy.ndarray as input
class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``
    """


    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation


    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be scaled.

        Returns:
            np.ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)


    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):

        self.size = size

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be cropped.

        Returns:
            CV Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    """Crop the given CV Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be cropped and resized.

        Returns:
            np.ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4)
                                                    for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4)
                                                    for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)