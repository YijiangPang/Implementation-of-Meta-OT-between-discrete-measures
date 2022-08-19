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


#\cite(torchattacks.PGD)
class PGD_Attack_Exp(Attack):
    def __init__(self, model, noise_func, noise_batch_size, eps=0.3, alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.noise_func = noise_func
        self.noise_batch_size = noise_batch_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        if images.shape[0]*self.noise_batch_size > 2048:
            img_list = []
            images_split = torch.split(images, split_size_or_sections = int(2048/self.noise_batch_size))
            labels_split = torch.split(labels, split_size_or_sections = int(2048/self.noise_batch_size))
            for img_sub, lab_sub in zip(images_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(images, labels)

    def _forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        #expend the images and labels over noise_batch_size
        shape = torch.Size([images.shape[0], self.noise_batch_size]) + images.shape[1:]
        images_exp = images.unsqueeze(1).expand(shape)
        images_exp = images_exp.reshape(torch.Size([-1]) + images_exp.shape[2:])
        shape_exp = images_exp.shape
        images_exp, noise_added = self.noise_func(images_exp.view(len(images_exp), -1), flag_noise = True)
        images_exp = images_exp.view(shape_exp)
        noise_added = noise_added.view(shape_exp)

        shape = torch.Size([labels.shape[0], self.noise_batch_size])
        labels_exp = labels.unsqueeze(1).expand(shape)
        labels_exp = torch.flatten(labels_exp)



        if self._targeted:
            target_labels = self._get_target_label(images_exp, labels_exp)

        loss = nn.CrossEntropyLoss()

        adv_images = images_exp.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels_exp)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images_exp, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images_exp + delta, min=0, max=1).detach()

        #cal the mean grad over noise_batch_size
        shape = torch.Size([-1]) + torch.Size([self.noise_batch_size]) + delta.shape[1:]
        delta = delta.reshape(shape)
        delta = torch.mean(delta, dim = 1)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


#\cite(torchattacks.PGDL2)
class PGDL2_Exp(Attack):
    def __init__(self, model, noise_func, noise_batch_size, eps=1.0, alpha=0.1, steps=40, random_start=True, eps_for_division=1e-10):
        super().__init__("PGDL2", model)
        self.noise_func = noise_func
        self.noise_batch_size = noise_batch_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self._supported_mode = ['default', 'targeted']


    def forward(self, images, labels):
        if images.shape[0]*self.noise_batch_size > 2048:
            img_list = []
            images_split = torch.split(images, split_size_or_sections = int(2048/self.noise_batch_size))
            labels_split = torch.split(labels, split_size_or_sections = int(2048/self.noise_batch_size))
            for img_sub, lab_sub in zip(images_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(images, labels)


    def _forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)


        #expend the images and labels over noise_batch_size
        shape = torch.Size([images.shape[0], self.noise_batch_size]) + images.shape[1:]
        images_exp = images.unsqueeze(1).expand(shape)
        images_exp = images_exp.reshape(torch.Size([-1]) + images_exp.shape[2:])
        shape_exp = images_exp.shape
        images_exp, noise_added = self.noise_func(images_exp.view(len(images_exp), -1), flag_noise = True)
        images_exp = images_exp.view(shape_exp)
        noise_added = noise_added.view(shape_exp)

        shape = torch.Size([labels.shape[0], self.noise_batch_size])
        labels_exp = labels.unsqueeze(1).expand(shape)
        labels_exp = torch.flatten(labels_exp)



        if self._targeted:
            target_labels = self._get_target_label(images_exp, labels_exp)

        loss = nn.CrossEntropyLoss()

        adv_images = images_exp.clone().detach()
        batch_size = len(images_exp)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(adv_images.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels_exp)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images_exp
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images_exp + delta, min=0, max=1).detach()

        #cal the mean grad over noise_batch_size
        shape = torch.Size([-1]) + torch.Size([self.noise_batch_size]) + delta.shape[1:]
        delta = delta.reshape(shape)
        delta = torch.mean(delta, dim = 1)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


#\cite(torchattacks)
import time
from torchattacks.attack import Attack
from torchattacks import MultiAttack
from torchattacks import APGD
from torchattacks import APGDT
from torchattacks import FAB
from torchattacks import Square


class AutoAttack_exp(Attack):

    def __init__(self, model, noise_func, noise_batch_size, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False):
        super().__init__("AutoAttack", model)
        self.noise_func = noise_func
        self.noise_batch_size = noise_batch_size
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self._supported_mode = ['default']

        if version == 'standard':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
            ])
        else:
            raise ValueError("Not valid version. ['standard']")


    def forward(self, images, labels):
        if images.shape[0]*self.noise_batch_size > 2048:
            img_list = []
            images_split = torch.split(images, split_size_or_sections = int(2048/self.noise_batch_size))
            labels_split = torch.split(labels, split_size_or_sections = int(2048/self.noise_batch_size))
            for img_sub, lab_sub in zip(images_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(images, labels)


    def _forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #expend the images and labels over noise_batch_size
        shape = torch.Size([images.shape[0], self.noise_batch_size]) + images.shape[1:]
        images_exp = images.unsqueeze(1).expand(shape)
        images_exp = images_exp.reshape(torch.Size([-1]) + images_exp.shape[2:])
        shape_exp = images_exp.shape
        images_exp, noise_added = self.noise_func(images_exp.view(len(images_exp), -1), flag_noise = True)
        images_exp = images_exp.view(shape_exp)
        noise_added = noise_added.view(shape_exp)
        images_exp_bk = images_exp.clone().detach().to(self.device)

        shape = torch.Size([labels.shape[0], self.noise_batch_size])
        labels_exp = labels.unsqueeze(1).expand(shape)
        labels_exp = torch.flatten(labels_exp)

        #adv_images = self.autoattack(images, labels)
        adv_images_exp = self.autoattack(images_exp, labels_exp)


        #cal the mean grad over noise_batch_size
        shape = torch.Size([-1]) + torch.Size([self.noise_batch_size]) + adv_images_exp.shape[1:]
        adv_images_exp = adv_images_exp.reshape(shape)
        images_exp_bk = images_exp_bk.reshape(shape)
        delta = adv_images_exp - images_exp_bk
        delta = torch.mean(delta, dim = 1)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed


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