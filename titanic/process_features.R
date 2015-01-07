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

#Decision Trees -----------------
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

# Split back into Test and Train Sets
train <- combi[1:891,]
test <- combi[892:1309,]

# Build new tree with new features
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +  Title + FamilySize + FamilyID, 
             data=train, method="class")
fancyRpartPlot(fit)

#Make Prediction
Prediction <- predict(fit, test, type = "class")

#Submission File Output ----------------------------
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "submission_features.csv", row.names = FALSE)
