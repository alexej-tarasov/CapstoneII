#Library activation

library(dplyr)
library(caret)
library(tidyr)
library(plotly)
library(tidyverse)
library(lubridate)
library(broom)
library(ggplot2)
library(RCurl)
library(kableExtra)
library(e1071)
library(parallel)
library(doParallel)
library(rpart)
library(caTools)
library(Rborist)
library(randomForest)
library(gbm)

#In this section data will be read
#If you have any problem with the download of this dataset please download it from my github
#https://github.com/alexej-tarasov/CapstoneII


data <- read.csv("https://raw.githubusercontent.com/alexej-tarasov/CapstoneII/master/adult.csv")# please see comments above if you have any problem with download

#The data is separated to the test and validation set using the caret package.----
#Validation set will be 10 % percent of the total records in adult.csv

set.seed(1)
test_index <- createDataPartition(y = data$income, times = 1, p = 0.1, list = FALSE)
trainset <- data[-test_index,]
temp <- data[test_index,]
validation <- temp %>% 
  semi_join(trainset, by = "workclass") %>%
  semi_join(trainset, by = "education") %>%
  semi_join(trainset, by = "marital.status") %>%
  semi_join(trainset, by = "occupation") %>%
  semi_join(trainset, by = "relationship")%>%
  semi_join(trainset, by = "native.country")
accuracy_results<-data.frame()


#Below is shown numbers of records in training and validation set

print(paste("Trainset has ",count(trainset)," records"))
print(paste("Validation set has ",count(validation)," records"))




#Check the importance of the variables is shown below in the chart. Bigger values means more importance

model1<-rpart(income~.,data = trainset)
var1<-varImp(model1) # Calculation of iimportance
var1<-rownames_to_column(var1) # Transfer of the row names to the separate column in data frame
varImpotance<-var1[order(-var1$Overall),] # Reorder of the importances by value
ggplot(varImpotance,aes(x = reorder(rowname, Overall),y=Overall),fill = variable)+ #chart creation
  geom_bar(stat = "identity",fill="black")+
  xlab("Features")+ylab("Overall importance")+ggtitle("Features importance in dataset")+
  theme_minimal()+ theme(axis.text.x = element_text(angle = 90, hjust = 1))+coord_flip()

#Features workclass,sex, race, native.country,hours.per.week,age and fnlwgt has no significant importance to the prediction in and these features will be excluded from the analysis and save new feature optimised dataset as "trainset_opt"

trainset_opt<-trainset%>%select(-workclass,-sex, -race, -native.country,-fnlwgt,-hours.per.week,-age)
as_tibble(trainset_opt)%>%head(n=10)%>%kable() %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))


#For all training model was used the same method of cross validation with 5 folds with parallel processing on 7 cores of desktop
#Below first try is to use random forest algoriithm with parallel processing with parameter selection
#Duration of theh process with 7 parallel cores is ca 10 minutes


tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_rf<-train(income~., method="rf",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 
#model_rf # display of the model
ggplot(model_rf,highlight = T) # result of optimisation
prediction_rf<-predict(model_rf,validation)
#confusionMatrix(data=prediction_rf,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_rf<-confusionMatrix(data=prediction_rf,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="rf ",  
                                         Accuracy = accuracy_rf))
print(accuracy_results)
tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))

#Second model for the training was selected xgbTree method from caret package.----
#Method xgbTree is one of the methods of xgBoosting and is supposed for classification

getModelInfo()$xgbTree$type

#Calculation will take approximately 2 minutes with 7 cores

tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_xgbTree<-train(income~., method="xgbTree",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 

prediction_xgbTree<-predict(model_xgbTree,validation)
#confusionMatrix(data=prediction_xgbTree,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_xgbTree<-confusionMatrix(data=prediction_xgbTree,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="xgbTree ",  
                                         Accuracy = accuracy_xgbTree))
print(accuracy_results)
tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))

#Third methon was taken very popular method of xgbDart----
#Method xgbDART is XGBoost method and is supposed for classification

getModelInfo()$xgbDART$type

#Calculation will take approximately 16 minutes with 7 cores 

tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_xgbDART<-train(income~., method="xgbDART",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 
#model_xgbDART # display of the model
#ggplot(model_xgbDART,highlight = T)
prediction_xgbDART<-predict(model_xgbDART,validation)
#confusionMatrix(data=prediction_xgbDART,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_xgbDART<-confusionMatrix(data=prediction_xgbDART,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="xgbDART ",  
                                         Accuracy = accuracy_xgbDART))
print(accuracy_results)
tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))




#Training model with logistic regression (glm method in cated package)-----
#Method glm is logistic regression model and is supposed for classification

getModelInfo()$glm$type

#Calculation will take approximately 0.5 minutes with 7 cores 

tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_glm<-train(income~., method="glm",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 
prediction_glm<-predict(model_glm,validation)
#confusionMatrix(data=prediction_glm,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_glm<-confusionMatrix(data=prediction_glm,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="glm",  
                                         Accuracy = accuracy_glm))
print(accuracy_results)
tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))


#Training model with Generalized Boosted Regression Modeling-----
#Method gbm is Generalized Boosted Regression Modeling (GBM) and is supposed for classification

getModelInfo()$gbm$type

#Calculation will take approximately 1 minutes with 7 cores 

tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_gbm<-train(income~., method="gbm",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 
#model_gbm # display of the model
#ggplot(model_gbm,highlight = T)
prediction_gbm<-predict(model_gbm,validation)
#confusionMatrix(data=prediction_gbm,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_gbm<-confusionMatrix(data=prediction_rf,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="GBM ",  
                                         Accuracy = accuracy_gbm))
print(accuracy_results)
tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))

#Training model with LogitBoost-----
#LogitBoost is Boostet Logistic regression models and is supposed for classification

getModelInfo()$LogitBoost$type


#LogitBoost

tsCVstart<-Sys.time()

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)  # registration of the clusters for parallel processing
fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE) # adjusting control for allowing parrallel processing
set.seed(1)
model_LogitBoost<-train(income~., method="LogitBoost",data = trainset_opt,trControl = fitControl) # creation of the model for all attributes, take a lot of time 
#model_LogitBoost # display of the model
ggplot(model_LogitBoost,highlight = T)
prediction_LogitBoost<-predict(model_LogitBoost,validation)
#confusionMatrix(data=prediction_LogitBoost,reference=validation$income)
stopCluster(cluster)
registerDoSEQ()
accuracy_LogitBoost<-confusionMatrix(data=prediction_rf,reference=validation$income)$overall[["Accuracy"]]

accuracy_results <- bind_rows(accuracy_results,
                              data_frame(method="LogitBoost ",  
                                         Accuracy = accuracy_LogitBoost))

tsCVfinish<-Sys.time()

print(paste("Process duration ",trunc(unclass(tsCVfinish)-unclass(tsCVstart)),"sec"))

# Results section-----
#After evaluation of the methods rf, xgbTree, xgbDART,glm,LogitBoost and GBM the best model was selected. 
#The best model is xgbDart with the accuracy of 0.873. xgbDART is extreme Gradient boosting algorithm.alo
#All other used algorithm delivered also high accuracy which is  comparable to xgbDart and lies in the range of 0.85-0.86

print(accuracy_results)
