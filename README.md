# Non-negative matrix factorization and image classification

## Abstract
Automatic classification or clustering consists of partioning
 a set of objects (instances) described by a set of variables 
 into homogeneous groups (classes). With the advent of Big Data 
 and data science, clustering has become a very important task in 
 various fields including imaging. 
 The images are very widespread data especially on the web 
 and social networks (Instagram, Pinterest, Flickr, Google, etc ...). 
 The goal will be to propose a classification system for images from 
 various databases (photos , paintings, comics, etc.). 
 The non-negative matrix factorization makes it possible to approximate 
 a positive data matrix by the product of two matrices of lower 
 and positive dimensions. 
 By its simplicity, this method has become popular and is used both 
 in size reduction and also in clustering in a user-defined number 
 of classes k.

### Keywords


## Data Set
We used this data set from : [People Art](https://github.com/BathVisArtData/PeopleArt) 

## Example 
###### Explained variance for SepNMF with Spherical K-means using Norm L2 (rank = [20;70]).  

<p align="center"> 
<img src="https://github.com/mbenhamd/nmf-ter/blob/master/nmf_result/sepnmf-norm-2-skmeans--EVAR.png?raw=true">
</p>

## Conclusion

To conclude we will discuss the cases that we have not treated and the 
data set itself, clustering is a very complicated area because we 
travel in the vagueness as we have seen during this semester but we 
learned a lot about automatic classification.
First of all, we gave the different results and their explanation but 
that taught us several very useful things as data science students. 
Even with thoughtful pre-processing or without, ultimately the results 
were bad as well as a multitude of different combinations of the NMF 
(algorithms, methods, initializations) will not change the facts. 
The problem comes from the data set which is in itself not adapted 
and corresponds to the weak points of the NMF. The coefficients are 
positive numbers, but for every vector in the database, 
the amount of information is usually a small part that we use to 
reconstruct our points. Lines with too much variety in themselves 
do not allow the NMF to find a pattern.
In particular, it must be realized that while the NMF is widely used 
in science, its rigorous foundation has only been discovered for less 
than 30 years. At the moment we are writing and it is very likely that 
we have not yet found the best algorithm for this.
Then, we faced a lack of time to perform the analyzes with a RGB and 
binary matrix. You should know that we used two different services 
of Cloud Computing (Amazon Web Service, Google Cloud Computing)
, libraries Nimfa and NMF for R uses the CPU 
while for calculations Matrix intensive, a minimum is to use the GPU 
to drastically reduce the calculation time. All the resources 
(graphics, scripts) of the project are 
available [here](https://github.com/mbenhamd/nmf-ter).
Finally, since the data set was built for cross-depiction, 
we can say that the NMF as a technique is not suitable for this problem
 (other solutions exist such as convectional neural networks 
or Deep Semi-NMF.
