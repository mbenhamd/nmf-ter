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

#Sys.setenv(CUDA_PATH="/usr/local/cuda-9.1/")
#Sys.setenv(NMFGPU_ROOT="/usr/local/nmfgpu/") 


getwd()
setwd("/home/mbenhamd/Project/nmf-ter/")

data = read.csv("matrice-greyscale-25.csv")
dim(data)

d = as.matrix(as.data.frame(lapply(data, as.numeric)))
d = normalize(d, method = "range", range = c(0, 1), margin = 1L, on.constant = "quiet")
res <- nmfEstimateRank(d, seq(1,100), method='brunet', nrun=10, seed="nndsvd",.options='vtP')



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


# Use a seed that will be set before each first run
res <- nmf(t(d), 10, method='KL', nrun=2, seed="ica",.opt="v4tP",.pbackend=NULL)
plot(res)



data = read.csv("matrice-greyscale-16.csv")
d = as.matrix(as.data.frame(lapply(data, as.numeric)))
x16_sk_greyscale = fviz_nbclust(t(d), skmeans, k.max=100, method = "silhouette")
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

var = 1
bouh = "salut-"
test = paste(bouh,var,sep="")
W=1
paste("nmf-mu-",47,"-KmeansRandom-W.csv",sep="")

for (variable in seq(from=47,to=57)) {
  print(variable)
}


for (variable in seq(from=47,to=57)) {
  res = nmf(t(d),variable, algorithm="mu", initMethod="K-Means/Random", maxiter=10000)
  write.csv(res$W,file=paste("nmf-mu-",variable,"-KmeansRandom-W.csv",sep=""))
  write.csv(res$H,file=paste("nmf-mu-",variable,"-KmeansRandom-H.csv",sep=""))
  write.csv(res$RMSD,file=paste("nmf-mu-",variable,"-KmeansRandom-RMSD.csv",sep=""))
  write.csv(res$ElapsedTime,file=paste("nmf-mu-",variable,"-KmeansRandom-ElapsedTime.csv",sep=""))
  write.csv(res$Frobenius,file=paste("nmf-mu-",variable,"-KmeansRandom-Frobenius.csv",sep=""))
}


sparseness(d)


set.seed(567);

plot(0, xlim = c(1,10), ylim = c(0.4, 1.4), xlab = "Rank", ylab = "MSE")
cols <- c('deepskyblue', 'orange', 'firebrick1', 'chartreuse3');
for (col in cols) {
  index2 <- sample(which(!is.na(nsclc2)), 2000);
  nsclc3 <- nsclc2;
  nsclc3[index2] <- NA;
  err <- sapply(X = 1:10,
                FUN = function(k, d) {
                  z <- nnmf(d, k, verbose = FALSE);
                  mean((with(z, W%*%H)[index2] - nsclc2[index2])^2)
                },
                d = nsclc3
  );
  invisible(lines(err, col = col, type='b', lwd = 2, cex = 1));
}