library(devtools)
library(slam)
library(skmeans)
library(rngtools)
library(mclust)
library(raster)
library(schoolmath)
#library(nmfgpu4R)
library(NbClust)
library(ggplot2)
library(factoextra)
library(NMF)
library(FactoMineR)
library(skmeans)

Sys.setenv(CUDA_PATH="/usr/local/cuda-9.1/")
Sys.setenv(NMFGPU_ROOT="/usr/local/nmfgpu/") 


getwd()
setwd("/home/mbenhamd/Project/nmf-ter/")

data = read.csv("matrice-greyscale-32.csv")
dim(data)

d = as.matrix(as.data.frame(lapply(data, as.numeric)))


v = vector()
dyn.load("libKMCUDA.so")
for (variable in seq(from=10,to=100)){
  result = .External("kmeans_cuda", d,variable, tolerance=0.01, verbosity=0, average_distance=TRUE)  
  v=cbind(v,result$average_distance)  
}
which.min(v)


result = .External("kmeans_cuda", d, tolerance=0.01, metric="L2" ,verbosity=2, average_distance=TRUE)  
a = PCA(d)
a$eig

set.seed(123456)
V <- d

# Use a seed that will be set before each first run
res <- nmfEstimateRank(V, seq(10,15), method='brunet', nrun=1, seed=123456,.options='v4tp8')
plot(res)



data = read.csv("matrice-greyscale-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x16_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")
x16_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")

png('x16_k_greyscale.png',width = 1920, height = 1080)
plot(x16_k_greyscale)
dev.off()
png('x16_sk_greyscale.png',width = 1920, height = 1080)
plot(x16_sk_greyscale)
dev.off()

data = read.csv("matrice-binary-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x16_k_binary = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x16_sk_binary = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")

data = read.csv("matrice-greyscale-32.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x32_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x32_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")

png('x32_k_greyscale.png',width = 1920, height = 1080)
plot(x32_k_greyscale)
dev.off()
png('x32_sk_greyscale.png',width = 1920, height = 1080)
plot(x32_sk_greyscale)
dev.off()

data = read.csv("matrice-binary-32.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x32_k_binary = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x32_sk_binary = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")


data = read.csv("matrice-greyscale-64.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x64_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")
x64_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")

png('x64_k_greyscale.png',width = 1920, height = 1080)
plot(x64_k_greyscale)
dev.off()
png('x64_sk_greyscale.png',width = 1920, height = 1080)
plot(x64_sk_greyscale)
dev.off()

#Faire une NMG avec gpu pour skmeans anormalement faible

data = read.csv("matrice-greyscale-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))


res1 <- nmf(t(d), 47:57, method='KL', nrun=1, seed="ica",.options='v4tp8')
png('nmf-16-greyscale-47-57-KL-ica.png',width = 1920, height = 1080)
plot(res1)
dev.off()

res2 <- nmf(t(d), 47:57, method='KL', nrun=1, seed="random",.options='v4tp8')
png('nmf-16-greyscale-47-57-KL-random.png',width = 1920, height = 1080)
plot(res2)
dev.off()

data2 = read.csv("matrice-greyscale-32.csv")
d2 = as.matrix(as.data.frame(lapply(data2, as.numeric)))

res3 <- nmf(t(d2), 56:72, method='KL', nrun=1, seed="ica",.options='v4tp8')
png('nmf-32-greyscale-56-72-KL-ica.png',width = 1920, height = 1080)
plot(res3)
dev.off()

res4 <- nmf(t(d2), 56:72, method='KL', nrun=1, seed="random",.options='v4tp8')
png('nmf-32-greyscale-56-72-KL-random.png',width = 1920, height = 1080)
plot(res4)
dev.off()

res5 <- nmf(t(d2), 56:72, method='KL', nrun=1, seed="nndsvd",.options='v4tp8')
png('nmf-32-greyscale-56-72-KL-nndsvd.png',width = 1920, height = 1080)
plot(res5)
dev.off()

data64 = read.csv("matrice-greyscale-64.csv")
d64 = as.matrix(as.data.frame(lapply(data64, as.numeric)))
x64_k_greyscale = fviz_nbclust(t(d64), kmeans, k.max=256, method = "silhouette")
x64_sk_greyscale = fviz_nbclust(t(d64), skmeans, k.max=256, method = "silhouette")
png('x64_k_greyscale.png',width = 1920, height = 1080)
plot(x64_k_greyscale)
dev.off()
png('x64_sk_greyscale.png',width = 1920, height = 1080)
plot(x64_sk_greyscale)
dev.off()




