import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
# #############################################################################
# Load faces data
from PIL import Image
import numpy as np
from sklearn.decomposition import NMF


def image_to_array(filepath):
    im = Image.open(filepath).convert('L')
    (width, height) = im.size
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height, width))
    return greyscale_map


# s1 à s40
# 1 à 10

im = Image.open("orl_faces/s1/1.pgm")
im.show()
test = image_to_array("orl_faces/s1/1.pgm")
model = NMF(n_components=90, init='nndsvda', tol=5e-3)
W = model.fit_transform(test)
H = model.components_
A = model.inverse_transform(W)

Image.fromarray(A).show()
print(W)
