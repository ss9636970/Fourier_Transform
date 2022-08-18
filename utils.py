import matplotlib.pyplot as plt
import numpy as np

def plotpics(imgs:list, picNumber:list, figsize=[15, 4.5]):
    a, b = picNumber[0], picNumber[1]
    fig, ax = plt.subplots(a, b, figsize=figsize)
    for i in range(a):
        for j in range(b):
            if a == 1:
                p = ax[j]
            elif b == 1:
                p = ax[i]
            else:
                p = ax[i, j]
            
            index = b * i + j
            p.imshow(imgs[index], cmap='gray',interpolation='none')
    
    return fig