import tkinter as tk
from tkinter import filedialog, dialog
from PIL import ImageTk,Image
import numpy as np
import utils
import os
import time

class UI_Window:
    def __init__(self, name='UI'):
        self.name = name
        self.window = tk.Tk()
        self.filters = utils.Pic_Filters()

        self.window.title('image HW2')
        self.window.geometry('1200x700')
        self.canvas = tk.Canvas(self.window, width=1200, height=630, bg="white")
        self.canvas.place(x=0, y=70)

        self.oprate = tk.Menu(self.window)
        self.window.config(menu=self.oprate)

        self.menu1 = tk.Menu(self.oprate, tearoff=0)
        self.menu2 = tk.Menu(self.oprate, tearoff=0)
        self.menu3 = tk.Menu(self.oprate, tearoff=0)
        self.menu4 = tk.Menu(self.oprate, tearoff=0)

        self.menu1.add_command(label='開啟圖片', command=self.open_img)
        self.menu1.add_command(label='儲存圖片', command=self.save_img)

        self.menu2.add_command(label='histogram equalization', command=self.funsP1('histogram equalization'))
        self.menu2.add_command(label='local histogram equalization', command=self.funsP1('local histogram equalization'))
        self.menu2.add_command(label='Histogram matching', command=self.funsP1('Histogram matching'))

        self.menu3.add_command(label='Gaussian Filter', command=self.funsP2('Gaussian Filter'))
        self.menu3.add_command(label='Averaging Filter', command=self.funsP2('Averaging Filter'))
        self.menu3.add_command(label='Unsharp mask filter', command=self.funsP2('Unsharp mask filter'))
        self.menu3.add_command(label='Laplacian Filter', command=self.funsP2('Laplacian Filter'))
        self.menu3.add_command(label='Sobel Filter', command=self.funsP2('Sobel Filter'))
        self.menu3.add_command(label='Convolution-2', command=self.funsP2('Convolution-2'))

        self.menu4.add_command(label='Image denoising', command=self.funsP3('Image denoising'))
        self.menu4.add_command(label='Bilateral filter', command=self.funsP3('Bilateral filter'))
        self.menu4.add_command(label='NLM', command=self.funsP3('NLM'))
        self.menu4.add_command(label='NLM improvement', command=self.funsP3('NLM improvement'))

        self.oprate.add_cascade(label='檔案', menu=self.menu1)
        self.oprate.add_cascade(label='problem1', menu=self.menu2)
        self.oprate.add_cascade(label='problem2', menu=self.menu3)
        self.oprate.add_cascade(label='problem3~7', menu=self.menu4)

        self.img = None

    def open_img(self):
        imgPath = filedialog.askopenfile(mode='r')
        fileType = imgPath.name.split('.')[-1]
        if fileType == 'raw':
            img = np.fromfile(imgPath.name, dtype=np.uint8).reshape(512, 512)
            self.img = Image.fromarray(img)
        
        else:
            self.img = Image.open(imgPath.name)
    
        imgCanvas = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
        self.canvas.image = imgCanvas

    def save_img(self):
        imgPath = filedialog.asksaveasfile()
        path = imgPath.name
        self.img.save(path)

    def createNewWindow(self, filt):
        def fun():
            para = parameter.get(1.0, tk.END)
            para = para.split(',')
            para = list(map(float, para))
            print(para)
            newWindow.destroy()

            img = np.asarray(self.img).astype('float')
            img = filt(img, *para)
            self.img = self.filters.to_image(img)
            imgCanvas = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
            self.canvas.image = imgCanvas

        newWindow = tk.Toplevel(self.window)
        paralabel = tk.Label(newWindow, text='請輸入參數，用逗號分隔')
        paralabel.place(x=20, y=30)
        parameter = tk.Text(newWindow, width=22, height=1)
        parameter.place(x=10, y=10)
        bt = tk.Button(newWindow, text='確定', width=10, height=2, command=fun)
        bt.place(x=20, y=50)

    def funsP1(self, t):
        def fun():
            if t == 'histogram equalization':
                filt = self.filters.global_histogram_equalization
                para = {}
                img = np.asarray(self.img).astype('float')
                img = filt(img, **para)
                self.img = self.filters.to_image(img[0])
                imgCanvas = ImageTk.PhotoImage(self.img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
                self.canvas.image = imgCanvas

            if t == 'local histogram equalization':
                filt = self.filters.local_histogram_equalization
                self.createNewWindow(filt)

            if t == 'Histogram matching':
                preImage = Image.open('./data/test/flower.png')
                preImage = np.array(preImage).astype('uint8')
                filt = self.filters.histogram_matching
                para = {'histoPictureR': preImage}
                img = np.asarray(self.img).astype('float')
                img = filt(img, **para)
                self.img = self.filters.to_image(img[0])
                imgCanvas = ImageTk.PhotoImage(self.img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgCanvas)
                self.canvas.image = imgCanvas

        return fun

if __name__ == '__main__':
    window = UI_Window()
    window.run()