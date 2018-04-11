library(devtools)
library(slam)
library(skmeans)
library(rngtools)
library(mclust)
library(raster)
library(schoolmath)
library(nmfgpu4R)
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


library(parallel)

elbow <- function(min_max, frame) {
  set.seed(42)
  wss <- (nrow(frame)-1)*sum(apply(frame,2,var))
  for (i in min_max) {
    wss[i] <- sum(kmeans(frame,centers=i,algorithm = c('MacQueen'))$withinss)
  }
  return(wss)
}

parallel_elbow <- function(kmax, frame_choice){
# create separate kmin:kmax vectors 
cut_point <- 3
centers_vec <- 2:kmax    
x <- seq_along(centers_vec)
chunks <- split(centers_vec, ceiling(x/cut_point))
  
# use shared-memory parallelism on function of choice
results <- mclapply(chunks, FUN=elbow, frame=frame_choice)
  
# gather the results of each parallel run 
no_nas <- list()
for(i in 1:length(results)) { 
  no_nas[i] <- list(as.numeric(na.omit(results[[i]])))
}
  
vec <- unlist(no_nas)
final_vec <- setdiff(vec, vec[1])
final_vec <- append(vec[1],final_vec)
  
# create scree plot of all wss values
plot(1:length(final_vec), final_vec, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares", pch = 16, main="Elbow Plot", col="steelblue")
}


data = read.csv("matrice-greyscale-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x32_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")
x16_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")

data = read.csv("matrice-binary-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x16_k_binary = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x16_sk_binary = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")

data = read.csv("matrice-greyscale-32.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x32_k_greyscale = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x32_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")

data = read.csv("matrice-binary-32.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x32_k_binary = fviz_nbclust(t(d), kmeans, k.max=100, method = "silhouette")
x32_sk_binary = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")










