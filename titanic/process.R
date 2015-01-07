library(dplyr)
library(ggplot2)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Load data
train <- read.csv("./data//train.csv")
test <- read.csv("./data//test.csv")

head(train)
.Last.value %>% View() #View more data

# Examine Gender structure
str(train)
summary(train$Sex)
prop.table(table(train$Sex, train$Survived))
prop.table(table(train$Sex, train$Survived), 1) #gives you proportion row-wise
prop.table(table(train$Sex, train$Survived), 2) #gives you proportion column-wise

# Examine Class patterns
summary(train$Age)
train$Child <- 0
train$Child[train$Age < 18] <- 1
aggregate(Survived ~ Child + Sex, data=train, FUN=length) #total number
aggregate(Survived ~ Child + Sex, data=train, FUN=sum) #target ~ variables
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)}) #proportion
#head(train$Child, n=100)

#Examine Fare patterns - categorizing
summary(train$Fare)
train$Fare2 <- '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
train$Fare2[train$Fare < 10] <- '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# No. of people survived
table(train$Survived)
prop.table(table(train$Survived))

# Create new column in test and set prediction
test$Survived <- 0 #Assign all zero
test$Survived[test$Sex == 'female'] <- 1 #Assign female 1
test$Survived[test$Sex == 'female' & test$Fare == 3 & test$Fare >= 20] <- 1
head(test)

# Create submission dataframe and output to file
submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "submission_Pclass.csv", row.names = FALSE)
