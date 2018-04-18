import matplotlib
matplotlib.use('Agg')
import nimfa
import os, sys
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
from sklearn.metrics import silhouette_score, confusion_matrix,adjusted_rand_score
from sklearn.preprocessing import scale
import nimfa


def getGapStat(data, k, origDisp, nrefs=3):
    refDisps = np.zeros(nrefs)
    for i in range(nrefs):
        # Create new random reference set
        randomReference = np.random.random_sample(size=data.shape)

        # Fit to it
        km = KMeans(k)
        km.fit(randomReference)

        refDisp = km.inertia_
        refDisps[i] = refDisp
    gap = np.log(np.mean(refDisps)) - np.log(origDisp)
    return gap

size = int(sys.argv[1])
methode = sys.argv[2]
transpose = int(sys.argv[3])
gap = True if len(sys.argv) == 4 else False

datas = np.loadtxt(open("matrice-binary-"+str(size)+".csv", "rb"), delimiter=",")
if transpose == 1:
	datas = scale(datas).T
if transpose == 2:
	datas = scale(datas)
if transpose == 3:
	datas = scale(datas.T)
if transpose == 4:
	datas = datas
if transpose == 5:
	datas = datas.T

max_range = int((datas.shape[0]*datas.shape[1]) / (datas.shape[0]+datas.shape[1]))


elbow = []
elbow_score = []
elbow_inertia = []
gaps = []
silhouettes = []

K = range(2,max_range,1)

print("run : ")
for k in K:
    print(str(k)+" / "+ str(max_range) +" -- rappel : taille : " +str(size)+" - method : "+str(methode)+" - t :"+str(transpose))

    if methode == "kmeans":
    	kmeanModel = KMeans(n_clusters=k, n_jobs = -1).fit(datas)
    if methode == "skmeans":
    	kmeanModel = SphericalKMeans(n_clusters=k).fit(datas)

    score = kmeanModel.score(datas)

    elbow_score.append(score)
    elbow_inertia.append(kmeanModel.inertia_)

    if size < 50:
    	elbow.append(sum(np.min(cdist(datas, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / datas.shape[0])

    if gap:
	    gap = getGapStat(datas, k, kmeanModel.inertia_)
	    gaps.append(gap)

    cluster_labels = kmeanModel.predict(datas)
    silhouette_avg = silhouette_score(datas, cluster_labels)
    silhouettes.append(silhouette_avg)



file_name = "-"+str(methode)+"-"+str(size)+"-transpose-"+str(transpose)+".png"

fig = plt.figure(figsize=(14, 10))
plt.figure(figsize=(14, 10))
plt.plot(K, elbow_score, 'bx-')
plt.xlabel('k')
plt.ylabel('elbow score')
plt.title('Elbow Score')
plt.savefig("result/plt-elbow-score"+file_name, dpi=fig.dpi)

fig = plt.figure(figsize=(14, 10))
plt.figure(figsize=(14, 10))
plt.plot(K, elbow_inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('elbow inertia')
plt.title('Elbow Inertia')
plt.savefig("result/plt-elbow-inertia"+file_name, dpi=fig.dpi)

if size < 50:
	fig = plt.figure(figsize=(14, 10))
	plt.figure(figsize=(14, 10))
	plt.plot(K, elbow, 'bx-')
	plt.xlabel('k')
	plt.ylabel('elbow euclidian')
	plt.title('Elbow euclidian')
	plt.savefig("result/plt-elbow-eucidian"+file_name, dpi=fig.dpi)

if gap :
	fig = plt.figure(figsize=(14, 10))
	plt.figure(figsize=(14, 10))
	plt.plot(K, gaps, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Gaps')
	plt.title('Gaps')
	plt.savefig("result/plt-gaps"+file_name, dpi=fig.dpi)

fig = plt.figure(figsize=(14, 10))
plt.figure(figsize=(14, 10))
plt.plot(K, silhouettes, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette')
plt.title('Silhouette')
plt.savefig("result/plt-silhouette"+file_name, dpi=fig.dpi)
