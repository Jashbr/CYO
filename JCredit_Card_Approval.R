
#Loading Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("caTools")
library(caTools)
library(data.table)
library(purrr)
library(formattable)

#Credit Approval Dataset
crdappsrc <- read.table("crx.data", sep = ",")
credit_app <- crdappsrc
as.datatable(formattable(head(credit_app)))

#Rename Columns
credit_app <- credit_app %>%
  dplyr::rename(
    "gender" = V1,
    "age" = V2,
    "debt" = V3,
    "married" = V4,
    "customer" = V5,
    "education" = V6,
    "ethnicity" = V7,
    "yearsemployed" = V8,
    "priordefault" = V9,
    "employmentstatus" = V10,
    "creditscore" = V11,
    "driverlicence" = V12,
    "citizenship" = V13,
    "zipcode" = V14,
    "income" = V15,
    "approvalstatus" = V16
  )
as.datatable(formattable(head(credit_app)))

#Structure
str(credit_app)

#Binary Conversion of Data
credit_app$gender	 <- factor(ifelse(credit_app$gender=="a",1,0))
credit_app$employmentstatus	 <- factor(ifelse(credit_app$employmentstatus=="t",1,0))
credit_app$priordefault<- factor(ifelse(credit_app$priordefault=="t",1,0))
credit_app$approvalstatus	 <- factor(ifelse(credit_app$approvalstatus=="+",1,0))

#Missing Data
summary(credit_app)
credit_app$age <- as.numeric(as.character(credit_app$age)) # replaces ? with NA
num_data	<-  select_if(credit_app, is.numeric)
round(cor(num_data,use="complete.obs"),3)
Agelm<-lm(age~yearsemployed, data=credit_app,na.action=na.exclude)
Agelm
missing<-which(is.na(credit_app$age))
formattable(credit_app[missing,])

#Predict Age
credit_app$age[missing]<- predict(Agelm,newdata=credit_app[missing,])
formattable(credit_app[missing,])

#SD
sigma = sd(num_data$age, na.rm=TRUE)
sigma

#Normalize Age
credit_app$AgeNorm<- (credit_app$age-mean(credit_app$age, na.rm=TRUE))/sigma
par(mfrow=c(1,2), oma=c(0,0,1,0))
hist(credit_app$age,main=NULL,xlab="Age",col="grey")
hist(credit_app$AgeNorm,main=NULL,xlab="AgeNorm",ylab=NULL,col="cadetblue")
title("Distribution of Age Before and After Normalization",outer=TRUE)

#Distribution of Age by Credit Approval Status
ggplot(credit_app) + 
  aes(approvalstatus,age) + 
  geom_boxplot(outlier.colour="red") +
  theme_bw() +
  coord_flip() +
  labs(title="Distribution of Age by Credit Approval Status")

#Distribution of AgeNorm by Credit Approval Status
ggplot(credit_app) + 
  aes(approvalstatus,AgeNorm) + 
  geom_boxplot(outlier.colour="red") +
  theme_bw() +
  coord_flip() +
  labs(title="Distribution of AgeNorm by Credit Approval Status")

#Normalize YearsEmployed
credit_app$YearsEmployedNorm <- scale(credit_app$yearsemployed,center=TRUE,scale=TRUE)
par(mfrow=c(1,2), oma=c(0,0,1,0))
hist(credit_app$yearsemployed,main=NULL,xlab="YearsEmployed",col="grey")
hist(credit_app$YearsEmployedNorm,main=NULL,xlab="YearsEmployedNorm",ylab=NULL,col="cadetblue")
title("Distribution of YearsEmployed Before and After Normalization", outer=TRUE)

#Distribution of Values Before and After Log Transformation
Temp<- data.frame(scale(log(credit_app[,c(3,8,11,15)]+1),center=TRUE))
names(Temp)<-c("DebtLog","YearsEmployedLog","CreditScoreLog","IncomeLog")
credit_app<-cbind(credit_app,Temp)
par(mfrow=c(1,2), oma=c(0,0,1,0))
hist(credit_app$YearsEmployedNorm,main=NULL,xlab="YearsEmployedNorm",col="grey")
hist(credit_app$YearsEmployedLog,main=NULL,xlab="YearsEmployedLog",ylab=NULL,col="cadetblue")
title("Distribution of Values Before and After Log Transformation",outer=TRUE)

#Generate Train & Test
set.seed(1)
split<- sample.split(credit_app$approvalstatus, SplitRatio=0.75)
Train<- subset(credit_app,split==TRUE)
Test <- subset(credit_app, split==FALSE)

#Logistic Regression
#Model 1
LogFit<- glm(approvalstatus~AgeNorm+DebtLog+YearsEmployedLog+CreditScoreLog+IncomeLog, 
             data=Train,family=binomial)
summary(LogFit)
LogPred<- predict(LogFit,newdata=Train, type="response")
table(predicted = LogPred>0.5,actual = Train$approvalstatus) 

#Model 2
LogFit2<- glm(approvalstatus~YearsEmployedLog+CreditScoreLog+IncomeLog, 
              data=Train,family=binomial)
summary(LogFit2)
LogPred2<- predict(LogFit2,newdata=Train, type="response")
table(predicted = LogPred2>0.5,actual = Train$approvalstatus)

#Apply the prediction on Test
LogPred3<-predict(LogFit2, newdata=Test,type="response")
table(predicted = LogPred3>0.5,actual = Test$approvalstatus)

#KNN
#K=1
knn_fit <- knn3(approvalstatus~YearsEmployedLog+CreditScoreLog+IncomeLog, 
                data=Train)
y_hat_knn <- predict(knn_fit, Train, type = "class")
confusionMatrix(y_hat_knn, Train$approvalstatus)

#K=3
knn_fit3 <- knn3(approvalstatus~YearsEmployedLog+CreditScoreLog+IncomeLog, 
                 data=Train,k=3)
y_hat_knn3 <- predict(knn_fit3, Train, type = "class")
confusionMatrix(y_hat_knn3, Train$approvalstatus)

#K=5
knn_fit5 <- knn3(approvalstatus~YearsEmployedLog+CreditScoreLog+IncomeLog, 
                 data=Train,k=5)
y_hat_knn5 <- predict(knn_fit5, Train, type = "class")
confusionMatrix(y_hat_knn5, Train$approvalstatus)

#The Optimal K
ks <- seq(3, 100, 2)
accuracy <- map_df(ks, function(k){
  fit <- knn3(approvalstatus~YearsEmployedLog+CreditScoreLog+IncomeLog, 
              data=Train, k = k)
  
  y_hat <- predict(fit, Train, type = "class")
  cm_train <- confusionMatrix(y_hat, Train$approvalstatus)
  train_error <- cm_train$overall["Accuracy"]
  
  tibble(train = train_error, test = 0)
})
ks[which.max(accuracy$train)]
max(accuracy$train)

#Validation
y_hat_knn_test <- predict(knn_fit3, Test, type = "class")
confusionMatrix(y_hat_knn_test, Test$approvalstatus)$overall["Accuracy"]

#Chi-Squared test
credit_app$ethnicity	<-ifelse(is.na(credit_app$ethnicity),"v",credit_app$ethnicity)
tbl<-credit_app %>%
  group_by(ethnicity) %>%
  dplyr::summarise(Freq=n(),
                   approvalstatus=sum(approvalstatus==1))
tbl
chisq.test(tbl[2:3])

