# Predict Blood Donations
Data Science 4 Good (Swiss)  
`r format(Sys.time(), '%B %d, %Y')`  



# Introduction
Last update Tuesday 24.10.2017 13:44:20 CEST.

## Load and Check Data

```r
#https://s3.amazonaws.com/drivendata/data/2/public/9db113a1-cdbe-4b1c-98c2-11590f124dd8.csv
train <- read.csv('data/TrainingData.csv', stringsAsFactors = F)
#https://s3.amazonaws.com/drivendata/data/2/public/5c9fa979-5a84-45d6-93b9-543d1a0efc41.csv
test  <- read.csv('data/TestData.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # bind training & test data
```
Even though datasets are not the same, test dataset doesn't have attribute which should be predicted, we have joined data together to get the complete overview.

```r
# check data
str(full)
```

```
## 'data.frame':	776 obs. of  6 variables:
##  $ X                          : int  619 664 441 160 358 335 47 164 736 436 ...
##  $ Months.since.Last.Donation : int  2 0 1 2 1 4 2 1 5 0 ...
##  $ Number.of.Donations        : int  50 13 16 20 24 4 7 12 46 3 ...
##  $ Total.Volume.Donated..c.c..: int  12500 3250 4000 5000 6000 1000 1750 3000 11500 750 ...
##  $ Months.since.First.Donation: int  98 28 35 45 77 4 14 35 98 4 ...
##  $ Made.Donation.in.March.2007: int  1 1 1 1 0 0 1 0 1 0 ...
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

# Missing Values

```r
summary(full)
```

```
##        X         Months.since.Last.Donation Number.of.Donations
##  Min.   :  0.0   Min.   : 0.000             Min.   : 1.000     
##  1st Qu.:187.8   1st Qu.: 3.000             1st Qu.: 2.000     
##  Median :375.5   Median : 7.000             Median : 4.000     
##  Mean   :374.2   Mean   : 9.454             Mean   : 5.558     
##  3rd Qu.:558.2   3rd Qu.:14.000             3rd Qu.: 7.000     
##  Max.   :747.0   Max.   :74.000             Max.   :50.000     
##                                                                
##  Total.Volume.Donated..c.c.. Months.since.First.Donation
##  Min.   :  250               Min.   : 2.00              
##  1st Qu.:  500               1st Qu.:16.00              
##  Median : 1000               Median :28.00              
##  Mean   : 1389               Mean   :34.42              
##  3rd Qu.: 1750               3rd Qu.:50.00              
##  Max.   :12500               Max.   :98.00              
##                                                         
##  Made.Donation.in.March.2007
##  Min.   :0.0000             
##  1st Qu.:0.0000             
##  Median :0.0000             
##  Mean   :0.2396             
##  3rd Qu.:0.0000             
##  Max.   :1.0000             
##  NA's   :200
```
From summary prespective none of the attributes has N/A values except unknown 200 observations in test dataset for attribute which we need to predict.

# Exploratory Analysis
## Training Dataset Correlation
Initial investigation brought us the information we have all data numerical, so we can take a look closely to them and see what's the correlation amongs them and to the attribute which we want to predict.

```r
chart.Correlation(full[,c(6,2:5)], histogram=TRUE, pch=19)
```

![](Predict_Blood_Donations_files/figure-html/full-correlation-1.png)<!-- -->

Looking closer at the results it's clear that:

- _Number.of.Donations_ and _Total.Volume.Donated..c.c.._ are 100 % dependent (which was expectable) so we can use one of them only (I would vote for _Number.of.Donations_)
- After eliminating _Total.Volume.Donated..c.c.._ we can see that strongest correlation to attribute which we want to predict (_Made.Donation.in.March.2007_) have attributes _Months.since.Last.Donation_ and _Number.of.Donations_ so lets use those two for building the model.


```r
full$Total.Volume.Donated..c.c.. <- NULL
train$Total.Volume.Donated..c.c.. <- NULL
test$Total.Volume.Donated..c.c.. <- NULL
```

# Feature Engineering
Question if there is possibility to create some new feature is always a part of any kind of machine learning work. 

## Average Months per Donation
Here is the simpliest one which came to my mind using all attributes (considering _Total.Volume.Donated..c.c.._ as equivalent to _Number.of.Donations_) it's _Average.Months.Per.Donation_ calculated as diff between _Months.since.First.Donation_ and _Months.since.Last.Donation_ and divided by _Number.of.Donations_.

```r
full$Average.Months.Per.Donation <- (full$Months.since.First.Donation - full$Months.since.Last.Donation) / full$Number.of.Donations
```

## Distance to Average
Inspired by Timothy (Data Science 4 Good).


```r
full$Distance.To.Average <- exp(-abs(full$Average.Months.Per.Donation - full$Months.since.Last.Donation))
```
## Features Check and Set


```r
chart.Correlation(full[,c(5,2,3,4,6,7)], histogram=TRUE, pch=19)
```

![](Predict_Blood_Donations_files/figure-html/features-check-1.png)<!-- -->

- _Average.Months.Per.Donation_: with no big surprise this feature doesn't helped us so much. It has strong correlation to _Months.since.First.Donation_ but weak to all other including _Made.Donation.in.March.2007_ which we want to predict.
- _Distance.To.Average_: on the other hand this feature correlates to _Made.Donation.in.March.2007_ which we want to predict and also _Months.since.Last.Donation_ and _Number.of.Donations_.


```r
train$Average.Months.Per.Donation <- (train$Months.since.First.Donation - train$Months.since.Last.Donation) / train$Number.of.Donations
test$Average.Months.Per.Donation <- (test$Months.since.First.Donation - test$Months.since.Last.Donation) / test$Number.of.Donations

train$Distance.To.Average <- exp(-abs(train$Average.Months.Per.Donation - train$Months.since.Last.Donation))
test$Distance.To.Average <- exp(-abs(test$Average.Months.Per.Donation - test$Months.since.Last.Donation))
```

# Outliers
Outliers are big topic and sooner or later there is the time to get rid of them to improve machine learning algorithm. Of course we don't want to remove them all and definitelly we cannot remove them from test data.

When investigating outliers and defining limits for it's filtration we need to take into account all the relevant attributes in training data and do for example boxplots. Let's prepare boxplots for _Number.of.Donations_, _Months.since.Last.Donation_, _Months.since.First.Donation_ and _Average.Months.Per.Donation_. Maybe we can later take into account also _Donator.Type_.


```r
qnt <- quantile(train$Number.of.Donations, probs=c(.25, .75))
H <- 1.5 * IQR(train$Number.of.Donations)
nod.min <- qnt[1] - H
nod.max <- qnt[2] + H

nod <- ggplot(data = train, mapping = aes(factor("train"), Number.of.Donations)) + 
       geom_boxplot() + geom_hline(yintercept = nod.min, color = "orange") + 
       geom_hline(yintercept = nod.max, color = "orange") + 
       geom_text(aes(x = 0.5, y = nod.min, label = nod.min), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       geom_text(aes(x = 0.5, y = nod.max, label = nod.max), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       labs(title = "Outliers Number.of.Donations", x = "Train data") + theme_bw()
```


```r
qnt <- quantile(train$Months.since.Last.Donation, probs=c(.25, .75))
H <- 1.5 * IQR(train$Months.since.Last.Donation)
mld.min <- qnt[1] - H
mld.max <- qnt[2] + H

mld <- ggplot(data = train, mapping = aes(factor("train"), Months.since.Last.Donation)) + 
       geom_boxplot() + geom_hline(yintercept = mld.min, color = "orange") + 
       geom_hline(yintercept = mld.max, color = "orange") + 
       geom_text(aes(x = 0.5, y = mld.min, label = mld.min), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       geom_text(aes(x = 0.5, y = mld.max, label = mld.max), hjust=-0.5, vjust=-1, size = 3, colour = "orange") +
       labs(title = "Outliers Months.since.Last.Donation", x = "Train data") + theme_bw()
```


```r
qnt <- quantile(train$Months.since.First.Donation, probs=c(.25, .75))
H <- 1.5 * IQR(train$Months.since.First.Donation)
mfd.min <- qnt[1] - H
mfd.max <- qnt[2] + H

mfd <- ggplot(data = train, mapping = aes(factor("train"), Months.since.First.Donation)) + 
       geom_boxplot() + geom_hline(yintercept = mfd.min, color = "orange") + 
       geom_hline(yintercept = mfd.max, color = "orange") + 
       geom_text(aes(x = 0.5, y = mfd.min, label = mfd.min), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       geom_text(aes(x = 0.5, y = mfd.max, label = mfd.max), hjust=-0.5, vjust=+1.5, size = 3, colour = "orange") +
       labs(title = "Outliers Months.since.First.Donation", x = "Train data") + theme_bw()
```


```r
qnt <- quantile(train$Average.Months.Per.Donation, probs=c(.25, .75))
H <- 1.5 * IQR(train$Average.Months.Per.Donation)
adm.min <- qnt[1] - H
adm.max <- qnt[2] + H

adm <- ggplot(data = train, mapping = aes(factor("train"), Average.Months.Per.Donation)) + 
       geom_boxplot() + geom_hline(yintercept = adm.min, color ="orange") + 
       geom_hline(yintercept = adm.max, color = "orange") + 
       geom_text(aes(x = 0.5, y = adm.min, label = adm.min), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       geom_text(aes(x = 0.5, y = adm.max, label = adm.max), hjust=-0.3, vjust=-1, size = 3, colour = "orange") +
       labs(title = "Boxplot Average.Months.Per.Donation", x = "Train data") + theme_bw()
```

Boxplot visualizations contains boundaries (calculated as 25 % and 75 % quantiles +/- 1.5x interquartile range) defined by orange color under and over which we could look for outliers. But not all of them we want to filter out, because as much we filter out as less we will have data for training. So, there has to be boundary for each attribute given by purple line. Those purple limits (if any) are then used for finding outliers which are summarized in following table.

```r
multiplot(nod, mld, mfd, adm, cols=2)
```

![](Predict_Blood_Donations_files/figure-html/outliers-summary-1.png)<!-- -->

```r
rm(nod, nod.min, nod.max, mld, mld.min, mld.max, mfd, mfd.min, mfd.max, adm, adm.min, adm.max, qnt, H)
```

Let's remove outliers from training data.

```r
# 11 outliers X ids
X <- train[(train$Number.of.Donations > 30 | train$Months.since.Last.Donation > 50 | train$Average.Months.Per.Donation > 25), "X"]
length(X)
```

```
## [1] 11
```

```r
# 23 outliers X ids
#X <- train[(train$Number.of.Donations > 30 | train$Months.since.Last.Donation > 50 | train$Average.Months.Per.Donation > 20), "X"]
#length(X)

train <- train[!(train$X %in% X),]
```

# Prediction
## Model Tuning
It's not good idea to blindly train the model on train data and then submit the prediction on test data. So, let's first tune it a bit.

### Data Preparation
We can use test data split them to testing and training set and try to figure out which model would be the best.

```r
set.seed(123) #reproducibility
ind <- createDataPartition(y = train$Made.Donation.in.March.2007, p = 0.75, list = F)
train.train <- train[ind,]
train.test <- train[-ind,]
```

### Tunning prediction SVM
First algorithm which was chosen is Support Vector Machine from e1071 package. It's necessary to evaluate model better and tune the parameters. Documentation is here https://cran.r-project.org/web/packages/e1071/e1071.pdf. Tips on practical use here: https://cran.ms.unimelb.edu.au/web/packages/e1071/vignettes/svmdoc.pdf from which was taken idea to use tune.svn function.


```r
set.seed(123) #reproducibility
obj <- tune.svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train.train, gamma = 2^(-1:1), cost = 2^(2:5), probability = T)
obj
```

```
## 
## Parameter tuning of 'svm':
## 
## - sampling method: 10-fold cross validation 
## 
## - best parameters:
##  gamma cost
##    0.5    4
## 
## - best performance: 0.2197485
```


```r
set.seed(123) #reproducibility
obj <- tune.svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train, cost = 2^(2:5), gamma = 2^(-1:1) , probability = T)
obj
```

```
## 
## Parameter tuning of 'svm':
## 
## - sampling method: 10-fold cross validation 
## 
## - best parameters:
##  gamma cost
##      1    4
## 
## - best performance: 0.2031286
```
With this result we can perform the prediction either with recommended parameters for cost and gamma or with empirically found:

```r
set.seed(123) #reproducibility
svm_model <- svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train.train, probability = T, gamma = 0.1, cost = 10)
pred <- predict(svm_model, train.test, probability = T)
```

### Tuning prediction RandomForest
Second algorithm based on discussion tips on DrivenData site under the cometition was RandomForest.


```r
set.seed(345)  #reproducibility
rf_model <- randomForest(factor(Made.Donation.in.March.2007) ~ Number.of.Donations + Months.since.Last.Donation + Months.since.First.Donation + Average.Months.Per.Donation + Distance.To.Average, data = train)

# Show model error
plot(rf_model, ylim=c(0,0.3))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
```

![](Predict_Blood_Donations_files/figure-html/rf-tune-train-prediction-1.png)<!-- -->

```r
# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))
```

```
## Warning: package 'bindrcpp' was built under R version 3.3.2
```

```r
# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 4, colour = 'white') +
  labs(x = 'Variables') + coord_flip() + theme_bw()
```

![](Predict_Blood_Donations_files/figure-html/rf-tune-train-prediction-2.png)<!-- -->


```r
set.seed(345)  #reproducibility
rf_model <- randomForest(factor(Made.Donation.in.March.2007) ~ Distance.To.Average, data = train.train, sampsize = 5)

pred <- predict(rf_model, train.test, type = "prob")[,"1"] # we want to predict positive outcome probability
```

## Model Evaluation
When we have binary classification problem we can simple caclulate accuracy and present cofusion matrix, but in this case when we calculate probability as output not the 1 or 0 class we need to evaluate it differently.

For such cases is calculated Logartimic loss which is quite opposite than accuracy which we are trying to maximize, this measure we are trying to minimize. With following formula for it's caluclation:

Log loss = −1/n ∑[yi log(ŷi) + (1 − yi) log(1 − ŷi)]

we can establish R funkction LogLossBinary (taken from here: https://www.r-bloggers.com/making-sense-of-logarithmic-loss/) which would be used for prediction evaluation. As it was mentioned above, we are trying to minimize this loss, so lower number is better :-).


```r
LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

LogLossBinary(train.test$Made.Donation.in.March.2007, pred)
```

```
## [1] 0.4866215
```

## Final Prediction
Once we have model with best algorithm option we can do the prediction on test data and submit it to DrivenData competition.

### Final Prediction with SVM

```r
set.seed(123)  #reproducibility
model <- svm(Made.Donation.in.March.2007 ~ Number.of.Donations + Months.since.Last.Donation, data = train, probability = T, gamma = 1, cost = 20)
pred <- predict(model, test, probability = T)
```

Evaluation notes about best features combination for prediction:

SVM Features Setup                                                              |Log Loss Evaluation |Log Loss DrivenData
--------------------------------------------------------------------------------|---------|---------
Number.of.Donations + Months.since.Last.Donation                                |0.9319239|
and gamma = 2, cost = 8                                                         |1.3473510|
and gamma = 1, cost = 4                                                         |0.8616842|0.8568
and gamma = 1, cost = 20                                                        |0.8366062|0.8535
Number.of.Donations                                                             |0.9653561|
Months.since.Last.Donation                                                      |0.9799749|
Months.since.First.Donation                                                     |0.9804339|
Number.of.Donations + Months.since.Last.Donation + Months.since.First.Donation  |1.8958200|

### Final Prediction with RandomForest

```r
set.seed(345)  #reproducibility
rf_model <- randomForest(factor(Made.Donation.in.March.2007) ~ Distance.To.Average, data = train, sampsize = 20)

pred <- predict(rf_model, test, type = "prob")[,"1"] # we want to predict positive outcome probability
```

Evaluation notes about best features combination for prediction:

RF Features Setup                                                                       |Log Loss Evaluation |Log Loss DrivenData
----------------------------------------------------------------------------------------|---------|---------
Average.Months.Per.Donation                                                             |3.3900730|
Average.Months.Per.Donation + Months.since.First.Donation                               |2.3819310|
Average.Months.Per.Donation + Months.since.First.Donation + Months.since.Last.Donation  |1.9348560|
Average.Months.Per.Donation + Months.since.First.Donation + Number.of.Donations         |1.9057520|
and sampsize = 100                                                                      |0.7061318|
and sampsize = 50                                                                       |0.6488936|
and sampsize = 20                                                                       |0.6171891|0.5007
and sampsize = 20 (removed 11 outliers)                                                 |0.5107301|0.4998
and sampsize = 20 (removed 23 outliers)                                                 |0.4451362|0.5002
and sampsize = 10                                                                       |0.5850539|0.5017
and sampsize = 10 (removed 11 outliers)                                                 |0.5211104|
and sampsize = 5                                                                        |0.5796463|
and sampsize = 5 (removed 11 outliers)                                                  |0.5471914|
Distance.To.Average                                                                     ||
and sampsize = 20                                                                       |0.6258972|
and sampsize = 20 (removed 11 outliers)                                                 |0.5088622|0.4368
and sampsize = 10                                                                       |0.5948609|
and sampsize = 5                                                                        |0.5892294|
and sampsize = 5 (removed 11 outliers)                                                  |0.4866215|
Distance.To.Average + Average.Months.Per.Donation                                       ||
and sampsize = 20                                                                       |0.6137495|
and sampsize = 10                                                                       |0.5882106|
and sampsize = 5                                                                        |0.5650826|
Distance.To.Average + Average.Months.Per.Donation + Months.since.First.Donation         ||
and sampsize = 5                                                                        |0.5720995|
Distance.To.Average + Average.Months.Per.Donation + Number.of.Donations                 ||
and sampsize = 5                                                                        |0.5586326|
Distance.To.Average + Number.of.Donations                                               ||
and sampsize = 5                                                                        |0.5580574|

# Write Output to File

```r
out <- data.frame(X = test$X, pred = pred)
names(out) <- c("","Made Donation in March 2007")
head(out, 5)
```

```
##       Made Donation in March 2007
## 1 659                       0.382
## 2 276                       0.100
## 3 263                       0.188
## 4 303                       0.066
## 5  83                       0.498
```

```r
write.csv(out, "data/BloodDonationSubmission.csv", row.names = F)
```
