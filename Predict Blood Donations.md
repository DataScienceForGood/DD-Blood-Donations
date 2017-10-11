---
title: "Predict Blood Donations"
author: "Data Science 4 Good (Swiss)"
date: "10/10/2017"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

library(ggplot2) # visualization
library(dplyr) # data manipulation
library(mice) # imputation
library(e1071) # svmlib
library(PerformanceAnalytics) # correlation
library(caret)
setwd("~/Dropbox/DrivenData/Predict Blood Donations")
```

## 1 Introduction

### 1.1 Load and Check Data
```{r load-data}
#https://s3.amazonaws.com/drivendata/data/2/public/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv
train <- read.csv('data/TrainingData.csv', stringsAsFactors = F)
#https://s3.amazonaws.com/drivendata/data/2/public/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv
test  <- read.csv('data/TestData.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data
```
Even though datasets are not the same, test dataset doesn't have attribute which should be predicted, we have joined data together to get the complete overview.
```{r check-data}
# check data
str(full)
```
We are working with 776 observations and 6 variables from which the last one is the one which we need to predict. We have got idea about what are the values in particular variables and that all of them are numeric.

Variable Name               |Description
----------------------------|--------------------------------------------------------
X                           |Id of record
Months.since.Last.Donation  |Number of monthis since this donor's most recent donation
Number.of.Donations         |Total number of donations that the donor has made
Total.Volume.Donated..c.c.. |Total amount of blood that the donor has donated in cubuc centimeters
Months.since.First.Donation |Number of months since the donor's first donation
Made.Donation.in.March.2007 |Probability that a donor made a donation in March 2007

## 2 Missing Values
```{r summary-data}
summary(full)
```
From summary prespective none of the attributes has N/A values except unknown 200 observations in test dataset for attribute which we need to predict.

## 3 Exploratory Analysis
### 3.1 Training Dataset Correlation
Initial investigation brought us the information we have all data numerical, so we can take a look closely to them and see what's the correlation amongs them and to the attribute which we want to predict.
```{r full-correlation, warning = FALSE}
chart.Correlation(full[,2:6], histogram=TRUE, pch=19)
```

Looking closer at the results it's clear that:

- _Number.of.Donations_ and _Total.Volume.Donated..c.c.._ are 100 % dependent (which was expectable) so we can use one of them only (I would vote for _Number.of.Donations_)
- After eliminating _Total.Volume.Donated..c.c.._ we can see that strongest correlation to attribute which we want to predict (_Made.Donation.in.March.2007_) have attributes _Months.since.Last.Donation_ and _Number.of.Donations_ so lets use those two for building the model.

## 4 Feature Engineering
Question if there is possibility to create some new feature is always a part of any kind of machine learning work. 

### 4.1 Average Donations per Month
Here is the simpliest one which came to my mind using all attributes (considering _Total.Volume.Donated..c.c.._ as equivalent to _Number.of.Donations_) it's _Avg.Donations.per.Month_ calculated as diff between _Months.since.First.Donation_ and _Months.since.Last.Donation_ and divided by _Number.of.Donations_.
```{r average-donations}
full$Avg.Donations.per.Month <- (full$Months.since.First.Donation - full$Months.since.Last.Donation) / full$Number.of.Donations

summary(full$Avg.Donations.per.Month)
```
In some cases this new feature is 0 which indicate people who donate blood just once.

Let's take a look closely on relation to our original attributes:
```{r avg-donations-correlation, warning = FALSE}
chart.Correlation(full[,2:7], histogram=TRUE, pch=19)
```

With no big surprise the new feature doesn't helped us so much. It has strong correlation to _Months.since.First.Donation_ but weak to all other including _Made.Donation.in.March.2007_ which we want to predict.

### 4.2 Donator types
The previous feature engineering wasn't successful so much. On the other hand correlation plot showed us that histogram of new feature is quite skewed. So, to make it better we can establish new feature based on previous one which will define groups of donator types. Let's define them and apply, but first take a look at histogram again.
```{r investigate-average-feature}
ggplot(data = full, mapping = aes(full$Avg.Donations.per.Month)) + geom_histogram(binwidth = 1) + theme_bw()
```

*TODO - need to be finished and both engineered features can be tested in the SVM model ;-)*

## 5 Prediction
### 5.1 Model Tuning
It's not good idea to blindly train the model on train data and then submit the prediction on test data. So, let's first tune it a bit.

#### 5.1.1 Data Preparation
We can use test data split them to testing and training set and try to figure out which model would be the best.
```{r data-preparation}
set.seed(123) #reproducibility
ind <- createDataPartition(y = train$Made.Donation.in.March.2007, p = 0.75, list = F)
train.train <- train[ind,]
train.test <- train[-ind,]
```

#### 5.1.2 Tunning prediction (SVM from e1071 package)
First algorithm which was chosen is Support Vector Machine from e1071 package. It's necessary to evaluate model better and tune the parameters. Documentation is here https://cran.r-project.org/web/packages/e1071/e1071.pdf

```{r tunnning-prediction}
set.seed(123) #reproducibility
model <- svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train.train, probability = T, cross = 10)
pred <- predict(model, train.test, probability = T)

head(pred)

# Evaluation notes about best features combination for prediction:
# Number.of.Donations + Months.since.Last.Donation                                 0.9319239
# Number.of.Donations                                                              0.9653561
# Months.since.Last.Donation                                                       0.9799749
# Months.since.First.Donation                                                      0.9804339
# Number.of.Donations + Months.since.Last.Donation + Months.since.First.Donation   1.89582
```

#### 5.1.3 Tuning prediction (another algorithm from another package)
We can try carret package for example and another model like logistic regression or such. Please establish another section that we keep info what has been used and how to not repeat the same mistakes ;-). And please set the seed for reproducibility as you can see it above.

### 5.2 Model Evaluation
When we have binary classification problem we can simple caclulate accuracy and present cofusion matrix, but in this case when we calculate probability as output not the 1 or 0 class we need to evaluate it differently.

For such cases is calculated Logartimic loss which is quite opposite than accuracy which we are trying to maximize, this measure we are trying to minimize. With following formula for it's caluclation:

Log loss = −1/n ∑[yi log(ŷi) + (1 − yi) log(1 − ŷi)]

we can establish R funkction LogLossBinary (taken from here: https://www.r-bloggers.com/making-sense-of-logarithmic-loss/) which would be used for prediction evaluation. As it was mentioned above, we are trying to minimize this loss, so lower number is better :-).

```{r model-evaluation}
LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

LogLossBinary(train.test$Made.Donation.in.March.2007, pred)
```

### 5.3 Final Prediction
Once we have model with best algorithm option we can do the prediction on test data and submit it to DrivenData competition. Let's do it with reasonable output since it's possible just 3 times a day and can be done only by dedicated member of group!
```{r final-prediction}
model <- svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train, probability = T, cross = 10)
pred <- predict(model, test, probability = T)
head(pred, 5)
```

## 6 Write output to file
```{r write-to-file}
out <- data.frame(X = test$X, pred = pred)
names(out) <- c("","Made Donation in March 2007")
head(out, 5)
write.csv(out, "data/BloodDonationSubmission.csv", row.names = F)
```