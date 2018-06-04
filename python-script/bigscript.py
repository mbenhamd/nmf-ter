import nimfa
import glob
import os, sys
import math
from PIL import Image
import scipy
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
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import scale, Normalizer,normalize
import nimfa
from sklearn import metrics


methode = sys.argv[1]
transpose = int(sys.argv[2])

labels = np.loadtxt(open("labels-25.csv", "rb"), delimiter=",")
datas = np.loadtxt(open("matrice-greyscale-25.csv", "rb"), delimiter=",")

if transpose == 1:
	datas = preprocessing.normalize(datas, norm='l2')
if transpose == 2:
	datas = datas


# no knowledgment 
elbow_inertia = []
gaps = []
silhouettes = []
# score
adjusted_rank_score = []  #ARI
normalized_mutual_info_score = [] #NMI equivalent v_measure_score
adjusted_mutual_info_score = [] #AMI
mutual_info_score = []


max_range = 70
K = range(20,max_range,1)

print("run : ")
for k in K:
	print(str(k)+" / "+ str(max_range) +" -- method : "+str(methode)+" - t :"+str(transpose))

	if methode == "kmeans":
		kmeanModel = KMeans(n_clusters=k, n_jobs = -1).fit(datas)
	if methode == "skmeans":
		kmeanModel = SphericalKMeans(n_clusters=k).fit(datas)

	elbow_inertia.append(kmeanModel.inertia_)


	cluster_labels = kmeanModel.predict(datas)
	silhouette_avg = silhouette_score(datas, cluster_labels)
	silhouettes.append(silhouette_avg)


	#adjusted_rand_score
	adjusted_rank_score.append( metrics.adjusted_rand_score(labels.astype(np.int32), cluster_labels)  )
	normalized_mutual_info_score.append( metrics.normalized_mutual_info_score(labels.astype(np.int32), cluster_labels) )
	adjusted_mutual_info_score.append( metrics.adjusted_mutual_info_score(labels.astype(np.int32), cluster_labels) )
	mutual_info_score.append( metrics.mutual_info_score(labels.astype(np.int32), cluster_labels)  )



file_name = "-"+str(methode)+"-transpose-"+str(transpose)+".png"

fig = plt.figure(figsize=(14, 10))
plt.plot(K, silhouettes, 'bx-')
plt.axvline(x=K[silhouettes.index(max(silhouettes))], linestyle='--')
plt.xlabel('k')
plt.ylabel('silhouettes')
plt.title('silhouettes')
plt.savefig("result2/plt-silhouette"+file_name, dpi=fig.dpi)


fig = plt.figure(figsize=(14, 10))
plt.figure(figsize=(14, 10))
plt.plot(K, elbow_inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('elbow inertia')
plt.title('Elbow Inertia')
plt.savefig("result2/plt-elbow-inertia"+file_name, dpi=fig.dpi)


fig = plt.figure(figsize=(14, 10))
plt.plot(K, adjusted_rank_score, '-bx', label='adjusted_rank_score')
plt.plot(K, normalized_mutual_info_score, '-rx', label='normalized_mutual_info_score')
plt.plot(K, adjusted_mutual_info_score, '-yx', label='adjusted_mutual_info_score')
plt.plot(K, mutual_info_score, '-mx', label='mutual_info_score')
plt.legend(loc='upper left')
plt.savefig("result2/plt-score"+file_name, dpi=fig.dpi)
