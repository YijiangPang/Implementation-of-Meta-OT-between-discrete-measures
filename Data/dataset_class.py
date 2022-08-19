import numpy as np
import torchvision
from torchvision import transforms as T


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, flag_train, cfg_m = None):
        self.flag_split = "train" if flag_train else "test"
        torchvision.datasets.MNIST.__init__(self, root=cfg_m.datasets_root, train = flag_train, download=True)
        data = self.data
        data = data.double()/255.
        data = data.reshape(-1, 784)
        data = data/data.sum(axis=1, keepdims=True)
        self.data = data
    
    def __getitem__(self, index: int):
        (id_a, id_b) = np.random.randint(0, len(self.data), 2)
        return 0, 0, self.data[id_a], self.data[id_b]


class CIFAR10_PAIR(torchvision.datasets.CIFAR10):
    def __init__(self, flag_train, cfg_m = None, id_c = [1, 1]):
        self.cfg_m = cfg_m
        self.flag_split = "train" if flag_train else "test"
        torchvision.datasets.CIFAR10.__init__(self, root=cfg_m.datasets_root, train = flag_train, download=True)
        self.transform_norm = TransformsNormTrainGray(size = cfg_m.img_size) if self.flag_split == "train" else TransformsNormTestGray(size = cfg_m.img_size)
        self.targets_a = [i for i, id in enumerate(self.targets) if id == id_c[0]]
        self.targets_b = [i for i, id in enumerate(self.targets) if id == id_c[1]]

    def __getitem__(self, index: int):
        id_a = np.random.choice(self.targets_a, size = 1)[0]
        id_b = np.random.choice(self.targets_b, size = 1)[0]
        img_a, img_b = self.data[id_a], self.data[id_b]

        img_a_t = self.transform_norm(img_a)
        img_b_t = self.transform_norm(img_b)
        img_a_flat = img_a_t.flatten().double()
        img_b_flat = img_b_t.flatten().double()
        img_a_flat = img_a_flat/img_a_flat.sum()
        img_b_flat = img_b_flat/img_b_flat.sum()
        return img_a_t, img_b_t, img_a_flat, img_b_flat


class TransformsNormTrainGray:
    def __init__(self, size):
        self.train_transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize(size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor()
        ])
    def __call__(self, x):
        return self.train_transform(x)


class TransformsNormTestGray:
    def __init__(self, size):
        self.test_transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize(size),
            T.ToTensor()
        ])
    def __call__(self, x):
        return self.test_transform(x)