import numpy as np
from PIL import Image

class Pic_Transformations:
    def __init__(self, name='transformations'):
        self.name = name
        self.K = 10

    def image_adj(self, inputs):
        mi = inputs.min()
        if mi < 0:
            outputs = inputs - mi
            ma = outputs.max()
        
        else:
            outputs = inputs
            ma = inputs.max()

        if ma > 255:
            outputs = outputs * (255 / ma)
            return outputs
        
        else:
            return inputs

    def to_image(self, array):
        img = array.astype('uint8')
        img = Image.fromarray(img)
        return img

    def log_transformation(self, inputs, c=110):
        outputs = np.log10(inputs + 1) * c
        return self.image_adj(outputs)

    def gamma_correction(self, inputs, gamma=0.5, A=1):
        maxPixel = inputs.max()
        outputs = inputs / maxPixel
        outputs = A * (outputs ** gamma)
        outputs = outputs * maxPixel
        return self.image_adj(outputs)

    def image_nagative(self, inputs, L=256):
        outputs = (L - 1) - inputs
        return self.image_adj(outputs)

    def bilinear_transform(self, inputs, size=[600, 1100]):
        img = inputs
        sizeOri = np.array(img.shape)
        size = np.array(size)
        adjLen = (sizeOri - 1) / (size - 1)
        height = np.arange(size[0]) * adjLen[0]
        width = np.arange(size[1]) * adjLen[1]
        meshH, meshW = np.meshgrid(height, width)
        meshH, meshW = meshH.transpose(), meshW.transpose()
        meshHI, meshWI = np.floor(meshH), np.floor(meshW)       # 轉化後像素點的整數部分(對應原圖左邊或上方的點)
        meshWL = meshW - meshWI         # x 的第一個元素
        meshHL = meshH - meshHI         # y 的第一個元素
        meshWR = 1 - meshWL             # x 的第二個元素
        meshHR = 1 - meshHL
        x = np.concatenate([np.expand_dims(meshWL, axis=2), np.expand_dims(meshWR, axis=2)], axis=2)
        x = np.expand_dims(x, axis=2)
        y = np.concatenate([np.expand_dims(meshHL, axis=2), np.expand_dims(meshHR, axis=2)], axis=2)
        y = np.expand_dims(y, axis=2)
        meshQW = np.concatenate([np.expand_dims(meshWI, axis=2), np.expand_dims(meshWI+1, axis=2)], axis=2).astype('long')   # index 要求 Q 用的
        meshQH = np.concatenate([np.expand_dims(meshHI, axis=2), np.expand_dims(meshHI+1, axis=2)], axis=2).astype('long')
        mh, mw = sizeOri[0] - 1, sizeOri[1] - 1
        meshQW, meshQH = np.clip(meshQW, a_max=mw, a_min=0), np.clip(meshQH, a_max=mh, a_min=0)
        meshQ = []   # 對應一個像素要計算的元途中四個點
        for i in range(1, -1, -1):
            for j in range(1, -1, -1):
                w = meshQW[:, :, i]
                h = meshQH[:, :, j]
                meshQ.append(np.expand_dims(img[h, w], axis=2))

        meshQ = np.concatenate(meshQ, axis=2)
        a, b, c = meshQ.shape
        meshQ = meshQ.reshape([a, b, 2, 2]).astype('float')
        temp = np.einsum('ijkl, ijlh->ijkh', x, meshQ)
        outputs = np.einsum('ijkl, ijfl->ijkf', temp, y).squeeze()
        return outputs.astype('uint8')

    def nearest_transform(self, inputs, size=[600, 1100]):
        img = inputs
        sizeOri = np.array(img.shape)
        size = np.array(size)
        adjLen = (sizeOri - 1) / (size - 1)
        height = np.arange(size[0]) * adjLen[0]
        width = np.arange(size[1]) * adjLen[1]
        meshH, meshW = np.meshgrid(height, width)
        meshH, meshW = meshH.transpose(), meshW.transpose()
        meshHI, meshWI = np.floor(meshH), np.floor(meshW)       # 轉化後像素點的整數部分(對應原圖左邊或上方的點)
        meshWL = ((meshW - meshWI) > 0.5) + 0.         # x 的第一個元素
        meshHL = ((meshH - meshHI) > 0.5) + 0.         # y 的第一個元素
        meshWR = 1 - meshWL             # x 的第二個元素
        meshHR = 1 - meshHL
        x = np.concatenate([np.expand_dims(meshWL, axis=2), np.expand_dims(meshWR, axis=2)], axis=2)
        x = np.expand_dims(x, axis=2)
        y = np.concatenate([np.expand_dims(meshHL, axis=2), np.expand_dims(meshHR, axis=2)], axis=2)
        y = np.expand_dims(y, axis=2)
        meshQW = np.concatenate([np.expand_dims(meshWI, axis=2), np.expand_dims(meshWI+1, axis=2)], axis=2).astype('long')   # index 要求 Q 用的
        meshQH = np.concatenate([np.expand_dims(meshHI, axis=2), np.expand_dims(meshHI+1, axis=2)], axis=2).astype('long')
        mh, mw = sizeOri[0] - 1, sizeOri[1] - 1
        meshQW, meshQH = np.clip(meshQW, a_max=mw, a_min=0), np.clip(meshQH, a_max=mh, a_min=0)
        meshQ = []   # 對應一個像素要計算的元途中四個點
        for i in range(1, -1, -1):
            for j in range(1, -1, -1):
                w = meshQW[:, :, i]
                h = meshQH[:, :, j]
                meshQ.append(np.expand_dims(img[h, w], axis=2))

        meshQ = np.concatenate(meshQ, axis=2)
        a, b, c = meshQ.shape
        meshQ = meshQ.reshape([a, b, 2, 2]).astype('float')
        temp = np.einsum('ijkl, ijlh->ijkh', x, meshQ)
        outputs = np.einsum('ijkl, ijfl->ijkf', temp, y).squeeze()
        return outputs.astype('uint8')


class Pic_Filters(Pic_Transformations):
    def __init__(self, name='Filters'):
        super(Pic_Transformations, self).__init__()
        self.name = name

    def extract_histogram(self, inputs):
        size = inputs.shape
        histogram = [0] * 256            # 儲存pixcel比例
        for i in range(256):
            histogram[i] = ((inputs == i)).sum()
        return histogram

    def global_histogram_equalization(self, inputs):
        inputs = inputs.astype('float')
        size = inputs.shape
        MN = size[0] * size[1]
        histogramBefore = self.extract_histogram(inputs)
        pixels = []
        temp = 0
        for i in range(256):
            temp += histogramBefore[i]
            pixels.append(temp)

        pixels2 = np.array(pixels) / MN
        pixels2 = (pixels2 * 255).round().astype('uint8')
        outputs = pixels2[inputs.astype('uint8')]
        histogramAfter = self.extract_histogram(outputs)
        return outputs, histogramBefore, histogramAfter

    # kernel 為 list ex: [3, 3]
    def local_histogram_equalization(self, inputs, kernel, k0=0.4, k1=0.02, k2=0.4, E=4):
        H, W = inputs.shape
        h = kernel
        w = kernel
        h2, w2 = int(h / 2), int(w / 2)
        boolean = np.zeros([H, W])
        gm, gv = inputs.mean(), inputs.std()
        for i in range(H):
            for j in range(W):
                t1, t2 = np.array([i - h2, i + h2]).clip(min=0, max=W), np.array([j - w2, j + h2]).clip(min=0, max=W)
                window = inputs[t1[0]:t1[1], t2[0]:t2[1]]
                lm, lv = window.mean(), window.std()
                # print(lm, lv)
                if lm <= k0 * gm and lv >= k1 * gv and lv <= k2 * gv:
                    boolean[i, j] = 1
        outputs = (inputs * (boolean * 3 + 1)).clip(min=0, max=256)
        return outputs

    # histoPictureR 為要當作 參考 histogram 的圖
    def histogram_matching(self, inputs, histoPictureR):
        inputs = inputs.astype('float')
        size = inputs.shape
        MN = size[0] * size[1]
        histoG = self.extract_histogram(inputs)
        histoR = self.extract_histogram(histoPictureR)

        pixels = [0] * 256            # 儲存pixcel比例
        temp = 0
        for i in range(256):
            temp += histoG[i]
            pixels[i] = temp
        pixelsG = np.array(pixels) / MN
        pixelsG = (pixelsG * 255).round().astype('uint8')

        pixels = [0] * 256            # 儲存pixcel比例
        temp = 0
        for i in range(256):
            temp += histoR[i]
            pixels[i] = temp
        pixelsR = np.array(pixels) / MN
        pixelsR = (pixelsR * 255).round().astype('uint8')

        pixelsRInv = [-1] * 256
        temp = []
        for i in range(256):
            ind = pixelsR[i]
            pixelsRInv[ind] = i
            temp.append(ind)

        temp = list(set(temp))
        temp = np.array(temp, dtype='int')
        for i in range(256):
            if pixelsRInv[i] == -1:
                c = np.argmin((temp - i) ** 2)
                ind = temp[c]
                pixelsRInv[i] = pixelsRInv[ind]

        pixelsRInv = np.array(pixelsRInv, dtype='uint8')
        outputs = pixelsRInv[pixelsG[inputs.astype('uint8')]]
        return outputs, histoR

    # size 為單一奇數
    def gaussian_filter(self, size, varience):
        s = int(size / 2)
        f = np.linspace((-1) * s, s, size)
        g, p = np.meshgrid(f, f)
        f = np.exp((-1) * (g ** 2 + p ** 2) / (2 * varience)) / (2 * np.pi * varience)
        f = f / f.sum()
        return f

    def average_filter(self, size):
        f = np.ones([size, size]) / (size ** 2)
        return f

    def unsharp_mask_filter(self, size, varience):
        gauF = self.gaussian_filter(size, varience)
        f = (-1) * gauF
        s = int((size - 1) / 2)
        f[s, s] += 1
        return f

    def laplacian_filter(self, size):
        def ff(n1, n2):
            c = 1
            for i in range(n1, n2+1):
                c *= i
            return c
        s = size - 1
        temp = []
        for i in range(0, size):
            c = (ff(i+1, s) / ff(1, s-i)) * ((-1) ** i)
            temp.append(c)
        temp = np.array(temp)
        p1 = np.zeros([size, size])
        p1[int(size / 2), :] = temp
        f = p1 + p1.T
        return f

    def sobel_filter(self, size, dir=1):   # 1為計算橫向，2為直向
        s = int(size / 2) + 1
        l = np.linspace(1, size, size)
        x, y = np.meshgrid(l, l)
        xy = np.concatenate([np.expand_dims(x, 0), np.expand_dims(y, 0)], axis=0)
        if dir == 1:
            dirs = np.array([1, 0])
        else:
            dirs = np.array([0, 1])

        center = np.array([int(size/2), int(size/2)]).reshape(2, 1, 1) + 1
        temp = (xy - center)
        distence = np.abs(temp).sum(axis=0).reshape(1, size, size)
        f = temp * dirs.reshape(2, 1, 1) / distence
        f = f.sum(axis=0)
        f[s-1, s-1] = np.array([0])
        return f * (size - 1)

    # 作業第三題
    def filterP3(self, num=1):
        if num == 1:
            outputs = np.zeros([3, 3])
            outputs[0, 0] = -1
            outputs[0, 2] = -1
            outputs[1, 1] = 6
            outputs[2, 0] = -1
            outputs[2, 2] = -1
        
        if num == 2:
            outputs = np.zeros([3, 3])
            outputs[0, 0] = 1
            outputs[0, 1] = 2
            outputs[0, 2] = 1
            outputs[1, 1] = 5
            outputs[2, 0] = 4
            outputs[2, 1] = 2
            outputs[2, 2] = 4
            outputs /= 25
        return outputs

    def padding(self, img, s):
        h, w = img.shape
        outputs = np.zeros([h+2 * s, w + 2 * s])
        outputs[s:-s, s:-s] = img
        return outputs

    def conv(self, filt, img):
        img = img.astype('float')
        h, w = img.shape
        size = filt.shape[0]
        s = int(size / 2)
        img = self.padding(img, s)
        outputs = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                x = np.array([i, i+2 * s + 1])
                y = np.array([j, j+ 2 * s + 1])
                imgf= img[x[0]:x[1], y[0]:y[1]]
                outputs[i, j] = (imgf * filt).sum()
        outputs = self.image_adj(outputs)
        return outputs
        
    def order_stat_filt(self,img, size=3, mode='median'):
        s = int(size / 2)
        img = img.astype('float')
        h, w = img.shape
        outputs = np.zeros([h, w])
        for i in range(h):
            for j in range(w):
                x = np.array([i-s, i+s+1]).clip(min=0, max=h+1)
                y = np.array([j-s, j+s+1]).clip(min=0, max=w+1)
                imgf= img[x[0]:x[1], y[0]:y[1]]
                if mode == 'median':
                    c = np.median(imgf)
                elif mode == 'min':
                    c = imgf.min()
                else:
                    c = imgf.max()
                outputs[i, j] = c
        return outputs

    def bila_filt(self, img, size=3, sigC=1, sigS=1):
        img = img.astype('float')
        varC = sigC ** 2
        varS = sigS ** 2
        s = int(size / 2)
        h, w = img.shape
        outputs = np.zeros([h, w])
        hl, wl = np.linspace(0, h-1, h), np.linspace(0, w-1, w)
        wi, hi = np.meshgrid(hl, wl)
        hi, wi = hi.reshape(h, w, 1), wi.reshape(h, w, 1)
        hw = np.concatenate([hi, wi], axis=2)
        for i in range(h):
            for j in range(w):
                x = np.array([i-s, i+s+1]).clip(min=0, max=h+1)
                y = np.array([j-s, j+s+1]).clip(min=0, max=w+1)
                imgV= img[x[0]:x[1], y[0]:y[1]]
                imgI = hw[x[0]:x[1], y[0]:y[1], :]
                cv, ci = img[i, j], hw[i:i+1, j:j+1, :]
                ind = (-1) * ((imgI - ci) ** 2).sum(axis=2) / (2 * varC)
                val = (-1) * np.abs(imgV - cv) / (2 * varS)
                d = np.exp(ind + val)
                u = d * imgV
                c = (u.sum() - cv) / (d.sum() - 1)
                # return d, u, c, cv
                outputs[i, j] = c
        outputs = self.image_adj(outputs)
        return outputs

    def non_local_mean_filt(self, img, Bratio, Sratio, sigma):
        img = img.astype('float')
        img = img / 255
        var = sigma ** 2
        BWSize = Bratio * 2 + 1
        SMSize = Sratio * 2 + 1
        h, w = img.shape

        imgPad = np.zeros([h + 2 * Bratio, w + 2 * Bratio])
        imgPad[Bratio:Bratio+h, Bratio:Bratio+w] = img
        imgPad[Bratio:Bratio+h, :Bratio] = np.flip(img[:, :Bratio], axis=1)
        imgPad[Bratio:Bratio+h, Bratio+w:] = np.flip(img[:, w-Bratio:], axis=1)
        imgPad[:Bratio, :] = np.flip(imgPad[Bratio:Bratio+Bratio, :], axis=0)
        imgPad[Bratio+h:, :] = np.flip(imgPad[h:h+Bratio, :], axis=0)
        N = SMSize ** 2

        outputs = np.zeros([h, w])
        for imgh in range(h):
            for imgw in range(w):
                imgPh, imgPw = imgh + Bratio, imgw + Bratio
                Swindow = imgPad[imgPh-Sratio:imgPh+Sratio+1, imgPw-Sratio:imgPw+Sratio+1]
                hp, wp = imgPh - Bratio, imgPw - Bratio

                weightSum = 0.
                valueSum = 0.
                for hi in range(hp, hp + BWSize - SMSize):
                    for wi in range(wp, wp + BWSize - SMSize):
                        Bwindow = imgPad[hi:hi + SMSize, wi:wi + SMSize]
                        similarty = np.sqrt(((Bwindow - Swindow) ** 2).sum()) / (N * 2 * var) * (-1)
                        weight = np.exp(similarty)
                        value = weight * imgPad[hi + Sratio, wi + Sratio]
                        valueSum += value
                        weightSum += weight

                c = valueSum / weightSum
                outputs[imgh, imgw] = c
        return outputs * 255

    # SINGLE-IMAGE DERAINING USING AN ADAPTIVE NONLOCAL MEANS FILTER
    def get_block(self, win, img):
        h, w, height, width = win
        return img[h:h+height, w:w+width]

    

    def get_bina(self, win, img, alpha):
        h, w, height, width = win
        imgO = self.get_block(win, img)
        winh1 = [h + 1, w, height, width]
        winh2 = [h - 1, w, height, width]
        winw1 = [h, w + 1, height, width]
        winw2 = [h, w - 1, height, width]
        imgH = (self.get_block(winh1, img) + self.get_block(winh2, img)) / 2
        imgW = (self.get_block(winw1, img) + self.get_block(winw2, img)) / 2
        imgC = (imgH + imgW) / 4
        outputs = (np.abs(imgO - imgC) > alpha) + 0.
        return outputs

    

    def non_local_mean_filt_imp(self, img, Bratio, Sratio, sigma=0.5, alpha=0.6):
        img = img.astype('float')
        img = img / 255
        var = sigma ** 2
        BWSize = Bratio * 2 + 1
        SMSize = Sratio * 2 + 1
        h, w = img.shape

        imgPad = np.zeros([h + 2 * Bratio, w + 2 * Bratio])
        imgPad[Bratio:Bratio+h, Bratio:Bratio+w] = img
        imgPad[Bratio:Bratio+h, :Bratio] = np.flip(img[:, :Bratio], axis=1)
        imgPad[Bratio:Bratio+h, Bratio+w:] = np.flip(img[:, w-Bratio:], axis=1)
        imgPad[:Bratio, :] = np.flip(imgPad[Bratio:Bratio+Bratio, :], axis=0)
        imgPad[Bratio+h:, :] = np.flip(imgPad[h:h+Bratio, :], axis=0)

        win = [Bratio, Bratio, h, w]
        binmap = self.get_bina(win, imgPad, alpha)

        imgPadmap = np.zeros([h + 2 * Bratio, w + 2 * Bratio])
        imgPadmap[Bratio:Bratio+h, Bratio:Bratio+w] = binmap
        imgPadmap[Bratio:Bratio+h, :Bratio] = np.flip(binmap[:, :Bratio], axis=1)
        imgPadmap[Bratio:Bratio+h, Bratio+w:] = np.flip(binmap[:, w-Bratio:], axis=1)
        imgPadmap[:Bratio, :] = np.flip(imgPadmap[Bratio:Bratio+Bratio, :], axis=0)
        imgPadmap[Bratio+h:, :] = np.flip(imgPadmap[h:h+Bratio, :], axis=0)
        # return imgPadmap[Bratio:Bratio+h, Bratio:Bratio+w] * 255
        outputs = np.zeros([h, w])
        for imgh in range(h):
            for imgw in range(w):
                imgPh, imgPw = imgh + Bratio, imgw + Bratio
                Swindow = imgPad[imgPh-Sratio:imgPh+Sratio+1, imgPw-Sratio:imgPw+Sratio+1]
                Smap = (-1) * (1 - imgPadmap[imgPh-Sratio:imgPh+Sratio+1, imgPw-Sratio:imgPw+Sratio+1])
        
                hp, wp = imgPh - Bratio, imgPw - Bratio

                weightSum = 1e-6
                valueSum = 0.
                for hi in range(hp, hp + BWSize - SMSize):
                    for wi in range(wp, wp + BWSize - SMSize):
                        Bwindow = imgPad[hi:hi + SMSize, wi:wi + SMSize]
                        Bmap = (-1) * (1 - imgPadmap[hi:hi + SMSize, wi:wi + SMSize])
                        
                        maps = Smap * Bmap
                        N = (maps == 1.).sum() + 1e-6

                        similarty = np.sqrt((((Bwindow - Swindow) * maps) ** 2).sum()) / (N * 2 * var) * (-1)
                        weight = np.exp(similarty)
                        value = weight * imgPad[hi + Sratio, wi + Sratio]
                        valueSum += value
                        weightSum += weight

                c = valueSum / weightSum
                outputs[imgh, imgw] = c

        return outputs * 255#, imgPadmap[Bratio:Bratio+h, Bratio:Bratio+w]
                




print('hii')

# def weight_get(self, win, img, k=0.1):
    #     imgwin = self.get_block(win, img)
    #     yMean = imgwin.mean()
    #     Wl = (1 + np.exp(-k * (imgwin - yMean))) ** (-1)
    #     return Wl

    # def cov_matrix(self, win, img):
    #     h, w, rediush, rediusw = win
    #     weight = self.weight_get(win, img).reshape(rediush * 2 + 1, rediusw * 2 + 1, 1)

    #     imgwin = self.get_block(win, img)
    #     imghplus = self.get_block([h+1, w, rediush, rediusw], img)
    #     imgwplus = self.get_block([h, w+1, rediush, rediusw], img)

    #     gradienth = (imghplus - imgwin).reshape(rediush * 2 + 1, rediusw * 2 + 1, 1)
    #     gradientw = (imgwplus - imgwin).reshape(rediush * 2 + 1, rediusw * 2 + 1, 1)
    #     g = gradienth * gradientw
    #     tempC = np.concatenate([gradienth ** 2, g, g, gradientw ** 2], axis=2)
    #     tempC = (weight * tempC).reshape(-1, 4)
    #     C = tempC.sum(axis=0).reshape(2, 2)
    #     Z = weight.sum()
    #     # print(C, Z)
    #     # print(C / Z)
    #     return C / Z

    # def get_bina(self, C, alpha=np.pi / 6, beta=0.1, gamma=0.1):
    #     U, S, V = np.linalg.svd(C)
    #     # print(U)
    #     # print(S)
    #     # print(V)
    #     thetap = np.arccos(U[0, 1])
    #     v = 255 ** 2
    #     lambp = S[1] / v
    #     mup = S[0] / v
    #     # a
    #     b = np.abs(lambp - mup)
    #     # print(b, mup)
    #     # print(b, mup)
    #     if b > beta and mup > gamma:
    #         return 1
    #     else:
    #         return 0

# def streak_map(self, img, rediush, rediusw):
    #     hpad, wpad = img.shape
    #     outputs = np.zeros([hpad, wpad])
    #     h, w = hpad - rediush * 2, wpad - rediusw * 2
    #     for i in range(rediush, h + rediush - 1):
    #         for j in range(rediusw, w + rediusw - 1):
    #             win = [i, j, rediush, rediusw]
    #             C = self.cov_matrix(win, img)
    #             c = self.get_bina(C)
    #             outputs[i, j] = c
    #     return outputs