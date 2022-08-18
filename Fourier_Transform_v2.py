import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

from BM3d.bm3d import run_bm3d
from guidedF.gf import guided_filter

class Fourier:
    def __init__(self, name='Foutier Transformer'):
        self.name = name

    def readRaw(self, path):
        pic = np.fromfile(path, 'uint8').reshape(512, 512)
        return pic.astype('float')

    def arrayToImage(self, a):
        outputs = a * 255
        outputs = Image.fromarray(outputs.astype('uint8'))
        return outputs

    def padding(self, img:np.array, size=[1024, 1024]):
        img = np.asarray(img, dtype=float)
        outputs = np.zeros(size)
        s = img.shape
        slic = [slice(i) for i in s]
        slic = tuple(slic)
        outputs[slic] = img
        return outputs

    def cutting(self, img:np.array, size=[512, 512]):
        img = np.asarray(img, dtype=float)
        slic = [slice(i) for i in size]
        slic = tuple(slic)
        return img[slic]

    #d表示對張亮的最後幾個維度做 shift
    def fft_shift(self, x:np.array, d=2):
        def shift(X, s):
            size = X.shape
            N = size[s]
            n = N // 2
            c = [i for i in range(len(X.shape))]
            c[s] = c[-1]
            c[-1] = s
            X = X.transpose(c).reshape(-1, N)
            outputs = np.zeros(size, dtype=complex).reshape(-1, N)
            outputs[:, :n] = X[:, n-1::-1]
            outputs[:, n:] = X[:, N:n-1:-1]
            return outputs.reshape(*size).transpose(c)
        
        size = x.shape
        l = len(size)
        for i in range(l-1, l-1-d, -1):
            x = shift(x, i)
        return x

    def FFT1d(self, x:np.array):
        x = np.asarray(x, dtype=complex)
        size = x.shape
        N = size[-1]
        x = x.reshape(-1, N)
        dataN = x.shape[0]

        if np.log2(N) % 1 > 0:
            raise ValueError("size of x must be a power of 2")

        N_min = min(N, 32)
        n = np.arange(N_min)
        k = n[:, None]
        M = np.exp(-2j * np.pi * n * k / N_min)
        X = np.einsum('ij, kjf->kif', M, x.reshape(dataN, N_min, -1))

        while X.shape[1] < N:
            s = int(X.shape[2] / 2)
            X_even = X[:, :, :s]
            X_odd = X[:, :, s:]
            factor = np.exp(-1j * np.pi * np.arange(X.shape[1]) / X.shape[1])

            temp = np.einsum('kij, i->kij', X_odd, factor)
            X = np.concatenate((X_even + temp, X_even - temp), axis=1)
        return X.reshape(*size)

    def invFFT1d(self, x:np.array):
        x = np.asarray(x, dtype=complex)
        size = x.shape
        N = size[-1]
        x = x.reshape(-1, N)
        dataN = x.shape[0]

        if np.log2(N) % 1 > 0:
            raise ValueError("size of x must be a power of 2")

        N_min = min(N, 32)
        n = np.arange(N_min)
        k = n[:, None]
        M = np.exp(2j * np.pi * n * k / N_min)
        X = np.einsum('ij, kjf->kif', M, x.reshape(dataN, N_min, -1))

        while X.shape[1] < N:
            s = int(X.shape[2] / 2)
            X_even = X[:, :, :s]
            X_odd = X[:, :, s:]
            factor = np.exp(1j * np.pi * np.arange(X.shape[1]) / X.shape[1])

            temp = np.einsum('kij, i->kij', X_odd, factor)
            X = np.concatenate((X_even + temp, X_even - temp), axis=1)
        return X.reshape(*size) / N

    # padding: padding後的圖片大小
    def FFT2d(self, x:np.array, padding=[1024, 1024]):
        if padding:
            x = self.padding(x, size=padding)
        x = np.ascontiguousarray(self.FFT1d(x))
        x = np.ascontiguousarray(self.FFT1d(x.transpose()))
        return x.transpose()

    # cutting: cutting後的圖片大小
    def invFFT2d(self, x:np.array, cutting=[512, 512]):
        x = np.ascontiguousarray(self.invFFT1d(x))
        x = np.ascontiguousarray(self.invFFT1d(x.transpose()))
        x = x.transpose()
        if cutting:
            x = self.cutting(x, size=cutting)
        return x
    
    # frequence domain histogram
    def FFThistogram(self, img):
        freqImg = self.FFT2d(img)
        freqImg = np.log(np.abs(freqImg))
        shiftFreqImg = self.fft_shift(freqImg, d=2)
        shiftFreqImg = np.abs(shiftFreqImg)

        columnW = freqImg.mean(axis=0)
        rowW = freqImg.mean(axis=1)
        
        cr = np.arange(0, columnW.shape[0])
        rr = np.arange(0, rowW.shape[0])

        fig, ax = plt.subplots(1, 5, figsize=(25, 4.5))
        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(freqImg, cmap='gray',interpolation='none')
        ax[2].imshow(shiftFreqImg, cmap='gray',interpolation='none')
        ax[3].plot(rr, rowW)
        ax[4].plot(cr, columnW)
        return fig

    # size 表示 filter 的大小 varience 代表常態分佈的變異數
    def gaussian_filter(self, size:list, varience:float):
        hs, ws = size
        h, w = np.arange(hs), np.arange(ws)
        y, x = np.meshgrid(h, w)
        y = y + 0.5 - hs / 2
        x = x + 0.5 - ws / 2

        f = np.exp((-1) * (y ** 2 + x ** 2) / (2 * varience)) / (2 * np.pi * varience)
        return f

    # size 表示 filter 的大小 ranges 代表取值的範圍大小
    def ideal_filter(self, size:list, ranges:int):
        hs, ws = size
        h, w = np.arange(hs), np.arange(ws)
        y, x = np.meshgrid(h, w)
        y = y + 0.5 - hs / 2
        x = x + 0.5 - ws / 2

        f = (np.abs(y) + np.abs(x)) < ranges
        return f + 0

    def bw_filter(self, size:list, n, D0:int):
        hs, ws = size
        h, w = np.arange(hs), np.arange(ws)
        y, x = np.meshgrid(h, w)
        y = y + 0.5 - hs / 2
        x = x + 0.5 - ws / 2

        D = np.sqrt(y ** 2 + x ** 2)
        f = 1 / (1 + (D / D0) ** (2 * n))
        return f

    # x 為 圖片張輛
    def low_pass(self, x, padding=[1024, 1024], mode='ideal', ranges=None, varience=None):
        x = np.asarray(x, dtype=float)
        orsize = x.shape
        freqBefore = self.FFT2d(x, padding)
        size = freqBefore.shape
        if mode == 'ideal':
            f = self.ideal_filter(size, ranges)
        elif mode == 'gaussian':
            f = self.gaussian_filter(size, varience)
        
        f = f / f.max()
        freqBeforeShift = self.fft_shift(freqBefore, d=2)
        freqAfterShift = f * freqBeforeShift
        freqAfter = self.fft_shift(freqAfterShift, d=2)
        outputs = self.invFFT2d(freqAfter, cutting=orsize)
        return outputs, freqBefore, freqAfter

    # x 為 圖片張輛
    def high_pass(self, x, padding=[1024, 1024], mode='ideal', ranges=None, n=None, D0=None):
        x = np.asarray(x, dtype=float)
        orsize = x.shape
        freqBefore = self.FFT2d(x, padding)
        size = freqBefore.shape
        if mode == 'ideal':
            f = self.ideal_filter(size, ranges)
        elif mode == 'butterworth':
            f = self.bw_filter(size, n, D0)
        
        f = f / f.max()
        f = 1 - f
        freqBeforeShift = self.fft_shift(freqBefore, d=2)
        freqAfterShift = f * freqBeforeShift
        freqAfter = self.fft_shift(freqAfterShift, d=2)
        outputs = self.invFFT2d(freqAfter, cutting=orsize)
        return outputs, freqBefore, freqAfter

    def conv(self, x, mode='gaussian', size=[7, 7], varience=50):
        x = np.asarray(x, dtype=float)
        if mode == 'gaussian':
            f = self.gaussian_filter(size, varience)
            f = f / f.sum()

        orsize = x.shape
        freqDx = self.FFT2d(x, padding=orsize)
        freqDf = self.FFT2d(f, padding=orsize)
        freqOutputs = freqDx * freqDf
        outputs = self.invFFT2d(freqOutputs, cutting=orsize)
        return outputs, freqOutputs

    # inputs 為圖片張量
    def inv_filter(self, x, padding=[1024, 1024], mode='gaussian', size=[7, 7], varience=50):
        x = np.asarray(x, dtype=float)
        orsize = x.shape
        freqDx = self.FFT2d(x, padding=padding)
        if mode == 'gaussian':
            f = self.gaussian_filter(size, varience)
            f = f / f.sum()
        
        s = x.shape
        freqDf = self.FFT2d(f, padding=padding)
        freqDf = 1 / freqDf
        freqReImg = freqDx * freqDf
        outputs = self.invFFT2d(freqReImg, cutting=orsize)
        return outputs, freqDx, freqDf, freqReImg

    def wiener_filter(self, x, padding=[1024, 1024], sn=1., sf=1., mode='gaussian', size=[7, 7], varience=50):
        x = np.asarray(x, dtype=float)
        orsize = x.shape
        freqDx = self.FFT2d(x, padding=padding)
        if mode == 'gaussian':
            f = self.gaussian_filter(size, varience)
            f = f / f.sum()
        
        s = x.shape
        freqDf = self.FFT2d(f, padding=padding)
        freqDf = freqDf / (freqDf * freqDf + sn / sf)
        freqReImg = freqDx * freqDf
        outputs = self.invFFT2d(freqReImg, cutting=orsize)
        return outputs, freqDx, freqDf, freqReImg

    # fast DCT
    def DCT1d(self, x):
        size = x.shape
        N = size[-1]
        x = x.reshape(-1, N)
        if N == 1:
            return x.reshape(*size)
        elif N == 0 or N % 2 != 0:
            raise ValueError()
        else:
            half = N // 2
            gamma = x[:, :half]
            delta = x[:, N-1:half-1:-1]
            alpha = self.DCT1d(gamma + delta)
            beta  = self.DCT1d((gamma - delta) / (np.cos(np.arange(0.5, half + 0.5) * (np.pi / N)) * 2.0))
            result = np.zeros_like(x)
            result[:, 0::2] = alpha
            result[:, 1::2] = beta
            result[:, 1:N-1:2] += beta[:, 1:]
            return result.reshape(*size)

    def invDCT1d(self, x, root=True):
        size = x.shape
        N = size[-1]
        if root:
            x = x.reshape(-1, N)
            x = x.copy()
            x[:, 0] = x[:, 0] / 2

        if N == 1:
            return x
        elif N == 0 or N % 2 != 0:
            raise ValueError()
        else:
            half = N // 2
            alpha = x[:, 0::2].copy()
            beta  = x[:, 1::2].copy()
            beta[:, 1:] += x[:, 1:N-1:2]
            alpha = self.invDCT1d(alpha, False)
            beta = self.invDCT1d(beta , False)
            beta /= np.cos(np.arange(0.5, half + 0.5) * (np.pi / N)) * 2.0
            x[:, :half] = alpha + beta
            x[:, N-1:half-1:-1] = alpha - beta

        if root:
            return x.reshape(*size) / N * 2
        else:
            return x.reshape(*size)

    def DCT2d(self, x:np.array, padding=[1024, 1024]):
        if padding:
            x = self.padding(x, size=padding)
        x = np.ascontiguousarray(self.DCT1d(x))
        x = np.ascontiguousarray(self.DCT1d(x.transpose()))
        return x.transpose()

    def invDCT2d(self, x:np.array, cutting=[512, 512]):
        x = np.ascontiguousarray(self.invDCT1d(x))
        x = np.ascontiguousarray(self.invDCT1d(x.transpose()))
        x = x.transpose()
        if cutting:
            x = self.cutting(x, size=cutting)
        return x

    def DCTfilter(self, x, padding=[1024, 1024], varience=20):
        x = np.asarray(x, dtype=float)
        orsize = x.shape
        freqBefore = self.DCT2d(x, padding)
        size = freqBefore.shape

        s = [size[0] * 2, size[1] * 2]
        f = self.gaussian_filter(s, varience)
        f = f[size[0]:, size[1]:]
        f = f / f.max()

        freqAfter = freqBefore * f
        
        outputs = self.invDCT2d(freqAfter, cutting=orsize)
        return outputs, freqBefore, freqAfter

    def bm3d(self, img, sigma=40):
        img = img.astype('float')
        im1, im2 = run_bm3d(img, sigma)
        return im1, im2

    def guideF(self, noisy, r=8, eps=0.05, s=4):
        noisy = noisy.astype('float') / 255
        I, _, _ = self.low_pass(noisy, padding=[1024, 1024], mode='gaussian', varience=np.exp(15))
        outputs = guided_filter(I, noisy, r=8, eps=0.05, s=4)
        return outputs * 255, I * 255
