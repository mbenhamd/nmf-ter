import matplotlib
matplotlib.use('Agg')
import nimfa
import scipy
import sys
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from spherecluster import SphericalKMeans
from sklearn.utils import shuffle
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import scale, Normalizer,normalize
from sklearn.decomposition import PCA, FastICA
from tqdm import tqdm


methode = sys.argv[1]
transpose = int(sys.argv[2])
seed = False if len(sys.argv) == 3 else sys.argv[3]

labels = np.loadtxt(open("labels-25.csv", "rb"), delimiter=",")
datas = np.loadtxt(open("matrice-greyscale-25.csv", "rb"), delimiter=",")

if transpose == 1:
	datas = preprocessing.normalize(datas, norm='l2')
if transpose == 2:
	datas = datas

max_iter=250
n_run=1
max_range = 23
K = range(20,max_range,1)
result = []


for k in tqdm(K):
	print(str(k)+" / "+ str(max_range) +" - method : "+str(methode)+" - norm :"+str(transpose)+" - seed :"+str(seed))

	options={ 'seed':'random_vcol', 'n_run':n_run, 'max_iter':max_iter, 'rank': k }

	if seed == "nndsvd":
		options['seed'] = "nndsvd"

	if seed == "pca" or seed == "ica" or seed == "skmeans" or seed == "kmeans":
		_data_dimension, _num_bases = datas.shape

		if seed == "pca":
			fica = PCA(n_components=k).fit(datas)
			H = fica.components_
		if seed == "ica":
			fica = FastICA(n_components=k).fit(datas)
			H = fica.components_
		if seed == "skmeans":
			skm = SphericalKMeans(n_clusters=k).fit(datas)
			H = skm.cluster_centers_
		if seed == "kmeans":
			skm = KMeans(n_clusters=k).fit(datas)
			H = skm.cluster_centers_

		W = np.random.random((_data_dimension, k))
		options['seed'] = None
		options['W'] = W
		options['H'] = H


	if methode == "lsnmf":
		init_nmf = nimfa.Lsnmf(datas, **options)

	if methode == "nmf":
		init_nmf = nimfa.Nmf(datas, **options)                            

	if methode == "sepnmf":
		init_nmf = nimfa.SepNmf(datas, **options)	

	if methode == "nsnmf":
		init_nmf = nimfa.Nsnmf (datas, **options)  


	res_nmf = init_nmf()

	result.append(res_nmf.summary())

	name = "nmf_result/"+str(methode)+"-norm-"+str(transpose)+"-"+str(seed)
	file_name = name+".png"

	W = init_nmf.basis()
	H = init_nmf.coef()
	np.save(name+"_W", W)
	np.save(name+"_H", H)



rss = [re['rss'] for re in result]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, rss, 'bx-')
plt.xlabel('k')
plt.ylabel('Residual Sum of Squares')
plt.title('RSS')
plt.savefig(name+"--RSS.png", dpi=fig.dpi)


evar = [re['evar'] for re in result]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, evar, 'bx-')
plt.xlabel('k')
plt.ylabel('evar')
plt.title('Explained variance')
plt.savefig(name+"--EVAR.png", dpi=fig.dpi)

coph = [re['cophenetic'] for re in result]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, coph, 'bx-')
plt.xlabel('k')
plt.ylabel('cophenetic')
plt.title('Cophenetic correlation coefficient of consensus matrix')
plt.savefig(name+"--COPHENETIC.png", dpi=fig.dpi)


sil = [metrics.silhouette_score(datas, np.array(res["predict_features"][0])[0]) for res in result ]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('silhouette')
plt.title('Silhouette score')
plt.savefig(name+"--SILHOUETTE.png", dpi=fig.dpi)


dispersions = [re['dispersion'] for re in result]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, dispersions, 'bx-')
plt.xlabel('k')
plt.ylabel('dispersions')
plt.title('Dispersion coefficient of consensus matrix')
plt.savefig(name+"--DISPERSION.png", dpi=fig.dpi)


adjusted_rank_score = [ metrics.adjusted_rand_score(
        labels.astype(np.int32), np.array(res["predict_features"][0])[0]
    ) for res in result ]
normalized_mutual_info_score = [ metrics.normalized_mutual_info_score(
        labels.astype(np.int32), np.array(res["predict_features"][0])[0]
    ) for res in result ]
adjusted_mutual_info_score = [ metrics.adjusted_mutual_info_score(
        labels.astype(np.int32), np.array(res["predict_features"][0])[0]
    ) for res in result ]
mutual_info_score = [ metrics.mutual_info_score(
        labels.astype(np.int32), np.array(res["predict_features"][0])[0]
    ) for res in result ]
fig = plt.figure(figsize=(14, 10))
plt.plot(K, adjusted_rank_score, '-bx', label='adjusted_rank_score')
plt.plot(K, normalized_mutual_info_score, '-rx', label='normalized_mutual_info_score')
plt.plot(K, adjusted_mutual_info_score, '-yx', label='adjusted_mutual_info_score')
plt.plot(K, mutual_info_score, '-mx', label='mutual_info_score')
plt.legend(loc='upper left')
plt.savefig(name+"--SCORE.png", dpi=fig.dpi)
