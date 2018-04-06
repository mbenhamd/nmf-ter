## 0. Create and register an MPI cluster
library(Rmpi)
library(doMPI)
cl <- startMPIcluster()
registerDoMPI(cl)

######################################################
library(devtools)
library(slam)
library(skmeans)
library(rngtools)
library(mclust)
library(magick)
library(sna)
library(raster)
library(imager)
library(schoolmath)
library(NMF)


getwd()
setwd("/home/mbenhamd/Project/nmf-ter/")


data = read.csv("matrice-greyscale.csv")
dim(data)
test = skmeans(as.matrix(data),k=49)
testmeans = kmeans(as.matrix(data),centers = 49)


qwerty = nmf(as.matrix(data),rank = 49,nrun = 5,H=test$prototypes)


blabla = as.matrix(data) 


show_images = function(data,from=1,to=10){
  final=vector()
for (variable in seq(from,to)) {
  b = matrix(data[variable,],nrow=32,ncol=32,byrow = T)
  final=cbind(final,b)  
}
  rst <- as.raster(final,max=255)
  plot(rst)
}

show_images(blabla,1,5)


# get matrix W
w <- basis(qwerty)
dim(w)
## [1] 200 3
# get matrix H
h <- coef(qwerty)
dim(h)
# The estimated target matrix can be retrieved via the generic method fitted, which returns a – generally big – matrix:
V.hat <- fitted(qwerty)
dim(V.hat)

BIC <- mclust::Mclust(as.matrix(data))
BIC$classification

      