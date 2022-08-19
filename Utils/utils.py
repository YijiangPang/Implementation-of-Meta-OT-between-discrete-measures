import torch
import torch.nn as nn
from torchattacks.attack import Attack
import numpy as np
import matplotlib.pyplot as plt
import os

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def insert(self, key, value):
        self[key] = value


def img_scale(im, im_shape):
    im = im if isinstance(im, np.ndarray) else im.data.cpu().numpy()
    im = im - im.min()
    im = im/im.max()
    im = np.reshape(im, im_shape)
    return im

def save_r(imgs, x_a, x_b, path, title):
    fig = plt.figure()
    fig.suptitle(title)
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs) + 2, i + 1 + 1)
        ax.imshow(im, cmap= 'gray', vmin=0, vmax=1) #'gray'
    ax = fig.add_subplot(1, len(imgs) + 2, 1)
    ax.imshow(img_scale(x_a, imgs[0].shape), cmap= 'gray', vmin=0, vmax=1) #'gray'
    ax = fig.add_subplot(1, len(imgs) + 2, len(imgs) + 2)
    ax.imshow(img_scale(x_b, imgs[0].shape), cmap= 'gray', vmin=0, vmax=1) #'gray'
    plt.show()
    plt.savefig(os.path.join(path, "OT_%s.png"%(title)))
    plt.close()

def save_r_cons(x_a, x_b, y_a, y_b, path, title):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(1, 4, 1)
    ax.axis('off')
    ax.imshow(x_a) 
    ax = fig.add_subplot(1, 4, 2)
    ax.axis('off')
    ax.imshow(x_b) 
    ax = fig.add_subplot(1, 4, 3)
    ax.axis('off')
    ax.imshow(y_a) 
    ax = fig.add_subplot(1, 4, 4)
    ax.axis('off')
    ax.imshow(y_b) 
    plt.show()
    plt.savefig(os.path.join(path, "OT_%s.png"%(title)))
    plt.close()