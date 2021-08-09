import pywt.data
import numpy as np
import torch
import cv2


class BGRToYCbCr(object):
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #TODO: YCbCr or YCrCb ?


class RGBToYCbCr(object):
    def __call__(self, img):
        return img.convert('YCbCr')


class PILToCHWarray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.moveaxis(img, -1, 0)
        return img


class WPT(object):
    def __init__(self, wavelet='db1', dtype='image', ch=3, lvl=2, debug=False):
        self.wavelet = wavelet
        self.dtype = dtype
        self.ch = ch
        self.lvl = lvl
        self.debug = debug

    def __call__(self, img):
        if self.dtype == 'image':
            img = np.array(img)
        # if self.ch == 1:
        #     wp = pywt.WaveletPacket2D(data=img[:, :, 0], wavelet=self.wavelet, mode='symmetric')
        #     data = [node.data for node in wp.get_level(self.lvl)]
        #     data = np.stack(data)
        #     return data
        # elif self.ch == 3:
        wps = [pywt.WaveletPacket2D(data=img[:, :, c], wavelet=self.wavelet, mode='symmetric', maxlevel=self.lvl) for c in
               range(self.ch)]
        data = np.stack([[node.data for node in wp.get_level(self.lvl)] for wp in wps])
        if self.debug:
            paths = np.stack([[node.path for node in wp.get_level(self.lvl)] for wp in wps])
            return data, paths
        return data

class WPTDownSample2(object):
    '''
    use wavelet 2d to downsample a tensor, only keeping the LL component
    '''
    def __init__(self, wavelet='db1'):
        self.wavelet = wavelet

    def __call__(self, img):
        # data = np.stack([pywt.dwt2(s, self.wavelet)[0] for s in img])
        data = np.stack([pywt.WaveletPacket2D(data=img[c, :, :], wavelet=self.wavelet, mode='symmetric', maxlevel=1).get_level(1)[0].data for c in range(img.shape[0])])
        return data


class WPTManual(object):
    def __init__(self, wavelet='db1', dtype='image', ch=3, lvl=2, msk=None, debug=False):
        self.wavelet = wavelet
        self.dtype = dtype
        self.ch = ch
        self.lvl = lvl
        self.msk = msk
        self.debug = debug

    def __call__(self, img):
        if self.dtype == 'image':
            img = np.array(img)

        def get_data(color_ch):
            wp = pywt.WaveletPacket2D(data=img[:, :, color_ch], wavelet=self.wavelet, mode='symmetric')
            return np.stack([wp[node].data for node in self.msk[color_ch]])

        return np.concatenate([get_data(i) for i in range(3)], axis=0)


class WPTReshape(object):
    """Merge first 2 dims"""

    def __call__(self, f):
        return np.concatenate([x for x in f])


class WPTMask(object):
    def __init__(self, msk=None):
        if isinstance(msk, np.ndarray):
            self.msk = msk
        else:
            self.msk = None
        self.reshaper = WPTReshape()

    def __call__(self, f):
        if self.msk is not None:
            return f[self.msk]
        else:
            return self.reshaper(f)


class WPTToTensor(object):
    def __call__(self, f):
        return torch.from_numpy(f).float()

class IdentityTransform(object):
    '''
    Simply do nothing :)
    '''
    def __call__(self, x):
        return x
