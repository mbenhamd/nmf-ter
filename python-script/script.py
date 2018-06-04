import glob
import random
import numpy
from PIL import Image
import nimfa
from sklearn.decomposition import NMF


def preprocess(V):
    print("Data preprocessing")
    V = (V - V.mean()) / numpy.sqrt(numpy.multiply(V, V).mean())
    V = numpy.maximum(numpy.minimum((V + 0.25) * 0.25, 1), 0)
    return V


def transform(images):
    X_data = numpy.array([])
    s = 0
    for i in images:
        image = Image.open(i).convert("L")
        arr = numpy.array(image)
        arr = arr.reshape(1, 10304)
        if X_data.any():
            X_data = numpy.concatenate((X_data, arr), axis=1)
        else:
            X_data = arr

        if s == 24:
            break
        s += 1
    return X_data


def compute(V):
    nmf = nimfa.Nmf(V, seed="random_vcol", rank=5, max_iter=500, sub_iter=10,
                    inner_sub_iter=10, beta=0.1, min_residuals=1e-8)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (nmf, nmf.seed, nmf.rank))
    fit = nmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
                - iterations: %d
                - final projected gradients norm: %5.3f
                - Euclidean distance: %5.3f 
                - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                                fit.distance(),
                                                                fit.distance(metric='kl'),
                                                                sparse_w, sparse_h))




images = glob.glob("dataset_exemples/s1/*.pgm")
random.shuffle(images, random.random)

matrice_Init = transform(images)
matrice_processed = NMF(matrice_Init)
W = matrice_processed.components_
H = matrice_processed.transform()

Image.fromarray(numpy.dot(W,H)).show()
