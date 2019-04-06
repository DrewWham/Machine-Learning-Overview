library(caret)
library(Matrix)
library(xgboost)
library(data.table)

#read in the pre processed data
train<-fread("./Supervised_example/train.csv")
test<-fread("./Supervised_example/test.csv")
#read.csv("Ftrain.csv")->train

#save the ID of each sample we will not use this in the training but will need it for plotting
train.ID<-train$Sample
train.ID<-as.character(train.ID)
train$Sample<-NULL
test.ID<-test$Sample
test.ID<-as.character(test.ID)
test$Sample<-NULL

#the Seq column contains the value we want to predict, saving this as the "Target" or Y-value
train$Target<-(train$Seq=="S. glynnii")*1
Target<-train$Target
train$Seq<-NULL
test$Seq<-NULL

#change the target value to 0 or 1 rather than the char strings
test$Target<-0
test$test<-1
train$test<-0
all.data<-rbind(train,test)


#we have several factors in the dataset, factors must be converted to "dummy" variables
dummies <- dummyVars(Target ~ ., data = all.data)
 all.data<-predict(dummies, newdata = all.data)
all.data<- data.frame(all.data)
test<-subset(all.data,test==1)
train<-subset(all.data,test==0)

train$test<-NULL
test$test<-NULL


#often '-9' is scored for genetic data when data is missing, we need to change that to NA
train[train== -9]<- NA
test[test== -9]<- NA


# xgboost requires sevral parameters they are set here, see: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
dtrain<- xgb.DMatrix(data=as.matrix(train),label=Target,missing=NA)
dtest<- xgb.DMatrix(data=as.matrix(test),missing=NA)
param <- list(  objective           = "binary:logistic", 
				        gamma                 =0.01,
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.01,
                max_depth           = 4,
                subsample           = 1,
                colsample_bytree    = 0.8
)


#because we have a small dataset we can use a large k-fold method to evaluate accuracy of the model on held out data
CV<-xgb.cv(data=dtrain,label=Target,params=param,nrounds=1000,nfold=53,missing=NA,prediction=T)
XGBm<-xgb.train( params=param,nrounds=1000,missing=NA,data=dtrain,label=Target)   
Prob<-predict(XGBm,newdata=dtest)

#we can access the held out data in the CV version here
Prob_cv<-CV$pred
mCV<-cbind(train.ID,Prob_cv)
mMod<-cbind(test.ID,Prob)
Model<-rbind(mCV,mMod)

#importance_matrix <- xgb.importance(names(train), model = XGBm,label=Target, data=dtrain)
importance_matrix <- xgb.importance(names(train), model = XGBm)
 xgb.plot.importance(importance_matrix)
 Model<-data.frame(Model)

#Here we will prep the CV predictions for plotting
Model$Prob_cv<-as.numeric(levels(Model$Prob_cv))[Model$Prob_cv]

names(Model)<-c("Sample","pS.trenchii")
Model$pS.glynnii<-1-Model$pS.trenchii
Model<-Model[!duplicated(Model$Sample),]
mModel<-melt(Model)
names(mModel)<-c("Sample","Species","Probability")
read.csv("./Supervised_example/PlotOrder.csv")->order


mModel$Sample <- factor(mModel$Sample, levels = mModel$Sample[match(order$Sample,mModel$Sample)])
mModel[!mModel$Sample=="NA",]->mModel
ggplot(mModel ,aes(x=Sample,y=Probability,fill=Species))+geom_bar(stat="identity",position="stack")+ scale_fill_brewer(palette="Spectral")+theme(axis.text.x = element_text(size=5, angle = 90, hjust = 1,colour="black"))

ggsave("./Supervised_example/Species_prob.pdf")
