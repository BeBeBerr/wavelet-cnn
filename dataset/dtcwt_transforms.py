import pydtcwt
import numpy as np
import torch
import cv2
import pytorch_wavelets

class DTCWT14Channel(object):
    '''
    ABS + Angle for 2 lowpasses and 6 highpasses, Y ; 2 channels for Cb Cr
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)
        self.dtcwt_transform_low = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=True)

    def transform(self, img):

        low_list = []
        high_list = []

        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, highs = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)

        low_list += [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]
        for i in range(highs[-1].shape[2]):
            high_real = highs[-1][0, 0, i][..., 0]
            high_imag = highs[-1][0, 0, i][..., 1]
            high_complex = high_real + high_imag * 1j
            high_abs = torch.abs(high_complex)
            # high_angle = torch.angle(high_complex)
            high_list += [high_abs]

        for c in range(1, 3):
            # Cb and Cr
            img_Y = np.array(img)[:, :, c]
            img_Y = torch.tensor(img_Y, dtype=torch.float32)
            img_Y = torch.unsqueeze(img_Y, 0)
            img_Y = torch.unsqueeze(img_Y, 0)
            low, highs = self.dtcwt_transform_low(img_Y)

            ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
            complex_1 = low_1r + low_1i * 1j
            complex_2 = low_2r + low_2i * 1j
            abs1 = torch.abs(complex_1)
            abs2 = torch.abs(complex_2)

            low_list += [abs1[0, 0], abs2[0, 0]]

        result = low_list + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)

class DTCWT18Channel(object):
    '''
    ABS + Angle for 2 lowpasses and 6 highpasses, Y ; 4 channels for Cb Cr
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)
        self.dtcwt_transform_low = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=True)

    def transform(self, img):

        low_list = []
        high_list = []

        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, highs = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)

        low_list += [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]
        for i in range(highs[-1].shape[2]):
            high_real = highs[-1][0, 0, i][..., 0]
            high_imag = highs[-1][0, 0, i][..., 1]
            high_complex = high_real + high_imag * 1j
            high_abs = torch.abs(high_complex)
            # high_angle = torch.angle(high_complex)
            high_list += [high_abs]

        for c in range(1, 3):
            # Cb and Cr
            img_Y = np.array(img)[:, :, c]
            img_Y = torch.tensor(img_Y, dtype=torch.float32)
            img_Y = torch.unsqueeze(img_Y, 0)
            img_Y = torch.unsqueeze(img_Y, 0)
            low, highs = self.dtcwt_transform_low(img_Y)

            ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
            complex_1 = low_1r + low_1i * 1j
            complex_2 = low_2r + low_2i * 1j
            abs1 = torch.abs(complex_1)
            abs2 = torch.abs(complex_2)
            angle1 = torch.angle(complex_1)
            angle2 = torch.angle(complex_2)

            low_list += [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]

        result = low_list + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)

class DTCWT30Channel(object):
    '''
    ABS + Angle for 2 lowpasses and 6 highpasses, Y Cb Cr
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)

    def transform(self, img):

        low_list = []
        high_list = []
        for c in range(3):
            img_Y = np.array(img)[:, :, c]
            img_Y = torch.tensor(img_Y, dtype=torch.float32)
            img_Y = torch.unsqueeze(img_Y, 0)
            img_Y = torch.unsqueeze(img_Y, 0)
            low, highs = self.dtcwt_transform(img_Y)

            ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
            complex_1 = low_1r + low_1i * 1j
            complex_2 = low_2r + low_2i * 1j
            abs1 = torch.abs(complex_1)
            abs2 = torch.abs(complex_2)
            angle1 = torch.angle(complex_1)
            angle2 = torch.angle(complex_2)

            low_list += [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]
            for i in range(highs[-1].shape[2]):
                high_real = highs[-1][0, 0, i][..., 0]
                high_imag = highs[-1][0, 0, i][..., 1]
                high_complex = high_real + high_imag * 1j
                high_abs = torch.abs(high_complex)
                # high_angle = torch.angle(high_complex)
                high_list += [high_abs]

        result = low_list + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)

class DTCWT48Channel(object):
    '''
    All channels
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)

    def transform(self, img):

        low_list = []
        high_list = []
        for c in range(3):
            img_Y = np.array(img)[:, :, c]
            img_Y = torch.tensor(img_Y, dtype=torch.float32)
            img_Y = torch.unsqueeze(img_Y, 0)
            img_Y = torch.unsqueeze(img_Y, 0)
            low, highs = self.dtcwt_transform(img_Y)

            ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
            complex_1 = low_1r + low_1i * 1j
            complex_2 = low_2r + low_2i * 1j
            abs1 = torch.abs(complex_1)
            abs2 = torch.abs(complex_2)
            angle1 = torch.angle(complex_1)
            angle2 = torch.angle(complex_2)

            low_list += [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]
            for i in range(highs[-1].shape[2]):
                high_real = highs[-1][0, 0, i][..., 0]
                high_imag = highs[-1][0, 0, i][..., 1]
                high_complex = high_real + high_imag * 1j
                high_abs = torch.abs(high_complex)
                high_angle = torch.angle(high_complex)
                high_list += [high_abs, high_angle]

        result = low_list + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)


class DTCWT16Channel(object):
    '''
    ABS + Angle for 2 lowpasses and 6 highpasses
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, highs = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)

        high_list = []
        for i in range(highs[-1].shape[2]):
            high_real = highs[-1][0, 0, i][..., 0]
            high_imag = highs[-1][0, 0, i][..., 1]
            high_complex = high_real + high_imag * 1j
            high_abs = torch.abs(high_complex)
            high_angle = torch.angle(high_complex)
            high_list += [high_abs, high_angle]

        result = [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]] + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)


class DTCWT10Channel(object):
    '''
    ABS + Angle for 2 lowpasses and 6 highpasses
    '''
    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, highs = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)

        high_list = []
        for i in range(highs[-1].shape[2]):
            high_real = highs[-1][0, 0, i][..., 0]
            high_imag = highs[-1][0, 0, i][..., 1]
            high_complex = high_real + high_imag * 1j
            high_abs = torch.abs(high_complex)
            # high_angle = torch.angle(high_complex)
            high_list += [high_abs]

        result = [abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]] + high_list
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)

class DTCWT7Channel(object):

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=False)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, highs = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        abs1 = torch.sqrt(torch.square(low_1r) + torch.square(low_1i))
        abs2 = torch.sqrt(torch.square(low_2r) + torch.square(low_2i))
        low = (abs1 + abs2) / 2
        low = low[0, ...] # squeeze

        high_abs = []
        for i in range(highs[-1].shape[2]):
            high_real = highs[-1][0, 0, i][..., 0]
            high_imag = highs[-1][0, 0, i][..., 1]
            abs = torch.sqrt(torch.square(high_real) + torch.square(high_imag))
            high_abs.append(abs)

        result = [low[0]] + high_abs
        result = torch.stack(result)
        return result

    def __call__(self, img):
        return self.transform(img)


class DTCWTLowPassAbsAngle(object):

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=True)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, _ = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)
        low_abs = (abs1 + abs2) / 2
        low_angle = (angle1 + angle2) / 2
        low = torch.stack([low_abs[0, 0], low_angle[0, 0]]) # squeeze

        return low

    def __call__(self, img):
        return self.transform(img)

class DTCWTLowPassAbsAngle4Channel(object):

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=True)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, _ = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        complex_1 = low_1r + low_1i * 1j
        complex_2 = low_2r + low_2i * 1j
        abs1 = torch.abs(complex_1)
        abs2 = torch.abs(complex_2)
        angle1 = torch.angle(complex_1)
        angle2 = torch.angle(complex_2)
        low = torch.stack([abs1[0, 0], abs2[0, 0], angle1[0, 0], angle2[0, 0]]) # squeeze

        return low

    def __call__(self, img):
        return self.transform(img)

class DTCWT1Channel(object):

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

        self.dtcwt_transform = pytorch_wavelets.DTCWTForward(J=nlevels, skip_hps=True)

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        img_Y = torch.tensor(img_Y, dtype=torch.float32)
        img_Y = torch.unsqueeze(img_Y, 0)
        img_Y = torch.unsqueeze(img_Y, 0)
        low, _ = self.dtcwt_transform(img_Y)

        ((low_1r, low_1i), (low_2r, low_2i)) = q2c(low)
        abs1 = torch.sqrt(torch.square(low_1r) + torch.square(low_1i))
        abs2 = torch.sqrt(torch.square(low_2r) + torch.square(low_2i))
        low = (abs1 + abs2) / 2
        low = low[0, ...] # squeeze

        return low

    def __call__(self, img):
        return self.transform(img)

def q2c(y, dim=-1):
    """
    Convert from quads in y to complex numbers in z.
    """

    # Arrange pixels from the corners of the quads into
    # 2 subimages of alternate real and imag pixels.
    #  a----b
    #  |    |
    #  |    |
    #  c----d
    # Combine (a,b) and (d,c) to form two complex subimages.
    y = y/np.sqrt(2)
    a, b = y[:,:, 0::2, 0::2], y[:,:, 0::2, 1::2]
    c, d = y[:,:, 1::2, 0::2], y[:,:, 1::2, 1::2]

    #  return torch.stack((a-d, b+c), dim=dim), torch.stack((a+d, b-c), dim=dim)
    return ((a-d, b+c), (a+d, b-c))

'''

class DTCWT1Channel(object):
    dtcwt_transform = pydtcwt.DTCWaveletTransform()

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

    def transform(self, img):
        img_Y = np.array(img)[:, :, 0]
        lowpasses, _ = DTCWT1Channel.dtcwt_transform.forward(img_Y, num_stages=self.nlevels)

        return np.expand_dims(lowpasses[0], 0)

    def __call__(self, img):
        return self.transform(img)
'''

class SimpleDTCWT(object):

    dtcwt_transform = pydtcwt.DTCWaveletTransform()

    def __init__(self, nlevels=2, channels=3):
        self.nlevels = nlevels
        self.channels = channels

    def transform(self, img):
        img = np.array(img)
        # perform dtcwt channelwisely
        slice_list = []
        for channel in range(self.channels):
            lowpasses, complex_highpasses = SimpleDTCWT.dtcwt_transform.forward(img[:, :, channel], num_stages=self.nlevels)
            
            slice_list.append(lowpasses[0]) # will give 4 lowpass images, only take the first one
            
            for level_idx in range(complex_highpasses.shape[0]):
                highpass = complex_highpasses[level_idx]
                for direction_idx in range(highpass.shape[0]):
                    resized_real = np.real(highpass[direction_idx])
                    resized_img = np.imag(highpass[direction_idx])

                    resize_time = self.nlevels - level_idx - 1 # how many times to resize, in order to have same size
                    
                    if resize_time > 0:
                        resized_real = SimpleDTCWT.dtcwt_transform.forward_lowpass_only(resized_real, resize_time)[0] # again, only take the 1st lowpass coefficient
                        resized_img = SimpleDTCWT.dtcwt_transform.forward_lowpass_only(resized_img, resize_time)[0]

                    complex_image = resized_real + resized_img * 1j
                    slice_list.append(np.abs(complex_image))
                    slice_list.append(np.angle(complex_image))
        
        # stack together
        stack = np.stack(slice_list)
        return stack

    def __call__(self, img):
        return self.transform(img)

class DTCWTToTensor(object):
    def __call__(self, f):
        return torch.from_numpy(f).float()



if __name__ == '__main__':
    SimpleDTCWT()(np.random.rand(256, 256, 3))