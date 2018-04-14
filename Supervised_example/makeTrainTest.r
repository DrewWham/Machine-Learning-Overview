library(xgboost)
library(caret)
library(reshape2)
library(dplyr)

#read in data
read.csv("Cell Size Clade D.csv")->Csize
read.csv("MiniC Clade D.csv")->MiniC
#read.csv("FullTrain.csv")->MiniC
read.csv("Supplemental Table 1.csv",stringsAsFactors=F)->metaData
read.csv("Clade D Master May 7.csv",stringsAsFactors=F)->Master

#get host data from metaData
vHost<-strsplit(metaData$Host[1:245],split=" ")
gHost<-lapply(vHost, `[[`, 1)
Host<-unlist(gHost)
dfHost<-data.frame(cbind(metaData$Sample[1:245],Host))
names(dfHost)[1]<-"Sample"
Master[,1:2]->Mlabs
merge(Mlabs,dfHost,by="Sample",all.x=T)->labs
labs$Host<-ifelse(labs$Host.x=="",as.character(labs$Host.y),labs$Host.x)
Master$Host<-NULL
Master<-merge(Master,labs[,c('Sample','Host')])

#get cell counts from cell count data
Master<-merge(Master,Csize,all.x=T)
#get MiniCircle data from file
Master<-merge(Master,MiniC,all.x=T)

#make ITS columns
Master$ITS_D1<- -9
Master$ITS_D4<- -9
Master$ITS_D6<- -9

Master$ITS_D1[Master$ITS.2.type=="D1-4-6"]<-1
Master$ITS_D4[Master$ITS.2.type=="D1-4-6"]<-1
Master$ITS_D6[Master$ITS.2.type=="D1-4-6"]<-1

Master$ITS_D1[Master$ITS.2.type=="D1-4"]<-1
Master$ITS_D4[Master$ITS.2.type=="D1-4"]<-1
Master$ITS_D6[Master$ITS.2.type=="D1-4"]<-0

Master$ITS_D1[Master$ITS.2.type=="D1"]<-1
Master$ITS_D4[Master$ITS.2.type=="D1"]<-0
Master$ITS_D6[Master$ITS.2.type=="D1"]<-0

msats<-Master[,c('Sample', 'Sym09','Sym09.1','Sym11', 'Sym11.1', 'Sym14', 'Sym14.1', 'Sym17','Sym17.1', 'Sym67', 'Sym67.1', 'Sym92', 'Sym92.1', 'Sym87', 'Sym87.1', 'Sym88', 'Sym88.1', 'Sym66', 'Sym66.1')]
msats[is.na(msats)]<-0
melt(msats,id.vars=c("Sample"))->mdata
mdata<-mdata[!is.na(mdata$value),]
mdata<-mdata[!mdata$value==0,]
strtrim(mdata$variable,5)->mdata$loci
mdata$variable<-NULL
dcast(mdata,Sample~loci+value,length)->castdata
castdata$uAlleles<-(apply(castdata[,2:ncol(castdata)], MARGIN = 1, FUN = function(x) sum((x>0)*1) ))

dropCol<-is.na(match(names(Master),c( 'Sym09','Sym09.1','Sym11', 'Sym11.1', 'Sym14', 'Sym14.1', 'Sym17','Sym17.1', 'Sym67', 'Sym67.1', 'Sym92', 'Sym92.1', 'Sym87', 'Sym87.1', 'Sym88', 'Sym88.1', 'Sym66', 'Sym66.1')))
sMaster<-Master[,dropCol]
Master<-merge(sMaster,castdata,all.x=T)
msats$nAlleles<-(apply(msats[,2:ncol(msats)], MARGIN = 1, FUN = function(x) sum((x>0)*1) ))
Master<-merge(Master,msats[,c('Sample','nAlleles')],all.x=T)
Master$HetroLoci<-Master$uAlleles-(Master$nAlleles/2)


Features<-c( "Sample", "Host", "Length", "Width", "ITS_D1", "ITS_D4" , "ITS_D6","HetroLoci","Seq"   )
Master<-Master[,Features]
dcast(mdata,Sample~loci,min,fill=0)->minAll
dcast(mdata,Sample~loci,max,fill=0)->maxAll
merge(minAll,maxAll,by="Sample",all=T)->Loci
Loci[Loci== Inf]<-NA
Loci[Loci== -Inf]<-NA

Master<-merge(Master,Loci,all.x=T)


train<-Master[!is.na(Master$Seq),]
test<-Master[is.na(Master$Seq),]

write.csv(train,"train.csv",row.names=F)
write.csv(test,"test.csv",row.names=F)
