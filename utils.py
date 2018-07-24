import scipy.misc as misc
import numpy as np


def resize_and_crop(img, size):
    if np.shape(img).__len__() < 3:
        img = np.dstack((img, img, img))
        h = np.size(img, 0)
        w = np.size(img, 1)

        if h < w:
            w = (512 * w) // h
            h = 512
            img = misc.imresize(img, [h, w], mode="RGB")
        else:
            h = (512 * h) // w
            w = 512
            img = misc.imresize(img, [h, w], mode="RGB")
        y = int(np.random.randint(0, h-size, 1))
        x = int(np.random.randint(0, w-size, 1))
        return img[y:y+size, x:x+size, :]
    else:
        img = img[:, :, :3]
        h = np.size(img, 0)
        w = np.size(img, 1)
        if h < w:
            w = (512 * w) // h
            h = 512
            img = misc.imresize(img, [h, w], mode="RGB")
        else:
            h = (512 * h) // w
            w = 512
            img = misc.imresize(img, [h, w], mode="RGB")
        y = int(np.random.randint(0, h - size, 1))
        x = int(np.random.randint(0, w - size, 1))
        return img[y:y + size, x:x + size, :]

def mapping(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255 / (max - min + 1e-10)

def preprocess(image, mode='BGR'):
    if mode == 'BGR':
        return image - np.array([103.939, 116.779, 123.68])
    else:
        return image - np.array([123.68, 116.779, 103.939])