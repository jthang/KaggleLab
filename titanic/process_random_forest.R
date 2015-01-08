library(dplyr)
library(ggplot2)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

# Load data
train <- read.csv("./data//train.csv")
test <- read.csv("./data//test.csv")

head(train)
.Last.value %>% View() #View more data

#Fitting by analzying data -------------------------------------------------------------------
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

#Decision Trees -----------------------------------------------------------------------------------------
#Create the gender model
fit <- rpart(Survived ~ Sex, train, method='class') #class is for categories. anova for continuous variable
fancyRpartPlot(fit)

#Building a deeper tree
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
plot(fit)
text(fit)
fancyRpartPlot(fit)

#Grow a full tree
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, 
             method="class", control=rpart.control(minsplit=2, cp=0))
plot(fit)
text(fit)
fancyRpartPlot(fit)

#Feature Engineering ------------------------------------------------------------------------
# Join test and train sets for feature engineering
test$Survived <- NA
combi <- rbind(train, test)

# Convert to a string
combi$Name <- as.character(combi$Name)
combi$Name[1]

# Split the name
strsplit(combi$Name[1], split='[,.]')
strsplit(combi$Name[1], split='[,.]')[[1]]
strsplit(combi$Name[1], split='[,.]')[[1]][2]

# Apply extraction formula to whole data
combi$Title <- sapply(combi$Name, FUN=function(x){strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
combi$Title[1]

#Combine small title groups together
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
table(combi$Title)

#Convert to a factor
combi$Title <- factor(combi$Title)
str(combi)

#Engineered Variable: Family Size
combi$FamilySize <- combi$SibSp + combi$Parch + 1

#Engineered Variable: Family
combi$Surname <- sapply(combi$Name, FUN=function(x){strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep='')
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

famIDs <- data.frame(table(combi$FamilyID))

#Filling in Missing Data -----------------------------------------------------------------------------

#Find where are the missing data
summary(combi)

# Using ANOVA to replace missing data
summary(combi$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

# Hardcode missing data
summary(combi$Embarked) #Found 2 missing data
which(combi$Embarked == '') #list which are the 2 missing data
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)

# Fill missing data with median
summary(combi$Fare)
which(combi$Fare == '')
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# Reduce No. of Factors ------------------------------------------------------------------------------------------
# R Random Forest can only take <32 levels, will have to reduce
str(combi)
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

# Split back into Test and Train Sets ----------------------------------------------------------------
train <- combi[1:891,]
test <- combi[892:1309,]

# Build Random Forest Ensemble
set.seed(415)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,
                    data=train, importance=TRUE, ntree=2000)

# Variable Importance
varImpPlot(fit)

#Make Prediction and Submission File Output ----------------------------------------------------------------------------
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)

#Using Conditional Inference Forests -----------------------------------------------------------------------------
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 

Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "submission_forest_ci.csv", row.names = FALSE)

