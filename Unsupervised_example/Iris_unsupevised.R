library(Rtsne)
library(ggplot2)
library(xgboost)

#read in the data
iris<-fread('iris.csv')

#for both of the unsupervised techniques duplicated observations will inhibit the analysis, be will start by removing them
unq_iris<-iris[!duplicated(iris[,1:4]),]

#principle component analysis will decompose the original dataset into principle components that contain all of the varience in the original dataset
pca<-prcomp(unq_iris[,1:4])

#we will need to extract just the principle components from the pca object, we will then make that a data_frame
iris_pca<-data.frame(pca$x)

#we will add the labels back to the dataset for the purpose of ploting
iris_pca$Species<-unq_iris$Species

#we can examin the pca's
summary(pca)

#a biplot will show the data ploted against the first 2 principle components, in red we can see the loading of the features loading on those axis
biplot(pca)

#we can plot the first 2 pca's and then label by speces to see how well this data dilimits the 3 species
ggplot(iris_pca,aes(x=PC1,y=PC2,col=Species))+geom_point()

ggsave("iris_pca.pdf")


#tsne is stocastic so in order to get a repeatable plot we must set R's random number generator
set.seed(7)

#this runs the tsne analysis, remeber that we removed duplicates above
tsne<-Rtsne(unq_iris[,1:4])

#here we gather just the resulting embedding coordinates and make that a data_frame
iris_tsne<-data.frame(tsne$Y)

#we add the original labels for plotting purposes
iris_tsne$Species<-unq_iris$Species

#change the names for plotting purposes
names(iris_tsne)[1:2]<-c('tsne1','tsne2')


#plot
ggplot(iris_tsne,aes(x=tsne1,y=tsne2,col=Species))+geom_point()

ggsave("iris_tsne.pdf")



