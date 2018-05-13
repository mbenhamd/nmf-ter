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
library(cluster)
library(rngtools)
library(NMF)
library(FactoMineR)
library(skmeans)


########## OpenCL Code #############
library(OpenCL)

p = oclPlatforms()  ## gets computing platform on server
d = oclDevices(p[[1]])  ## sets GPU device we want to use

print(p)
print(d)

#########################################"
getwd()
setwd("/home/mbenhamd/Project/nmf-ter/")

data = read.csv("matrice-greyscale-8.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))

x8_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=8, method = "silhouette")
x8_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=8, method = "silhouette")

