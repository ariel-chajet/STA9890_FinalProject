rm(list = ls())    #delete objects
cat("\014")
#install.packages("glmnet")
#install.packages("randomForest")
#install.packages("ggplot2")
library(glmnet)
library(randomForest)
library(ggplot2)


# import csv data into dataframe
rawdata.df = read.csv("CommAndCrime.csv")
dim(rawdata.df)


# drop non-predictive features and redundant predictors (used col index from original dataframe)
r.df = subset(rawdata.df, select = c(6:16,18,25:26,33,35:39,47:48,57,62:72,74:84,93:94,97:103,110:125,129,146:147))
dim(r.df)


# check null values
sum(is.na(r.df))


# loop over raw data to impute column means into missing values
for(j in 1:ncol(r.df)){
  r.df[is.na(r.df[,j]), j] = mean(r.df[,j], na.rm = TRUE)
}

sum(is.na(r.df))


# standardize predictors 
X = as.matrix(r.df[,-72]) # index to exclude response variable
y = as.matrix(r.df$ViolentCrimesPerPop) 

#par(mfrow=c(3,1))
#hist(y)
#hist(log(y))
#hist(sqrt(y))

y = sqrt(y)

n = dim(X)[1]
p = dim(X)[2]

mu.x = as.vector(apply(X, 2, "mean"))
sd.x = as.vector(apply(X, 2, "sd"))

for (i in c(1:n)){
  X[i,] = (X[i,] - mu.x)/sd.x
}


# create matrices for train and test residuals for each model
nsamples = 100

res.rid.tr = matrix(0, nrow = n*0.8, ncol = nsamples)
res.las.tr = matrix(0, nrow = n*0.8, ncol = nsamples)
res.elnet.tr = matrix(0, nrow = n*0.8, ncol = nsamples)
res.rf.tr = matrix(0, nrow = n*0.8, ncol = nsamples)

res.rid.ts = matrix(0, nrow = n*0.2, ncol = nsamples)
res.las.ts = matrix(0, nrow = n*0.2, ncol = nsamples)
res.elnet.ts = matrix(0, nrow = n*0.2, ncol = nsamples)
res.rf.ts = matrix(0, nrow = n*0.2, ncol = nsamples)

# create vectors for train and test R-squared for each model
Rsq.rid.tr = c(rep(0,nsamples))
Rsq.las.tr = c(rep(0,nsamples))
Rsq.elnet.tr = c(rep(0,nsamples))
Rsq.rf.tr = c(rep(0,nsamples))

Rsq.rid.ts = c(rep(0,nsamples))
Rsq.las.ts = c(rep(0,nsamples))
Rsq.elnet.ts = c(rep(0,nsamples))
Rsq.rf.ts = c(rep(0,nsamples))


#####################################################################
#Ridge Regression
#####################################################################


set.seed(5)


# for loop 100 times to fit model, calculate fit statistics, and store results
for (k in 1:nsamples){
  
# randomly split data into training and test set
  i.mix = sample(1:n)
  df.split = 1:(n*0.8)

  i.train=i.mix[df.split]
  i.test=i.mix[-df.split]

  X.train = X[i.train,]   # training predictors
  y.train = y[i.train]    # training responses
  X.test = X[i.test,] # test predictors
  y.test = y[i.test]  # test responses

  cv.rid.fit = cv.glmnet(X.train, y.train, alpha=0)

  rid.fit = glmnet(X.train, y.train, alpha=0, lambda=cv.rid.fit$lambda.min)
    
  y.train.hat = predict(rid.fit, newx = X.train, type = "response")   # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat = predict(rid.fit, newx = X.test, type = "response")     # y.test.hat=X.test %*% fit$beta  + fit$a0

  res.rid.tr[,k] = y.train - y.train.hat
  res.rid.ts[,k] = y.test - y.test.hat
  
  Rsq.rid.tr[k] = 1 - (  mean((y.train.hat - y.train)^2)  /  mean((y - mean(y))^2)  ) 
  Rsq.rid.ts[k] = 1 - (  mean((y.test.hat - y.test)^2)  /  mean((y - mean(y))^2)  )

  cat(sprintf("k=%3.f| Rsq.rid.ts=%.2f| Rsq.rid.tr=%.2f| \n", k,  Rsq.rid.ts[k],  Rsq.rid.tr[k]))
}


#####################################################################
#Lasso Regression
#####################################################################


set.seed(5)


# for loop 100 times to fit model, calculate fit statistics, and store results
for (k in 1:nsamples){
  
  # randomly split data into training and test set
  i.mix = sample(1:n)
  df.split = 1:(n*0.8)
  
  i.train=i.mix[df.split]
  i.test=i.mix[-df.split]
  
  X.train = X[i.train,]   # training predictors
  y.train = y[i.train]    # training responses
  X.test = X[i.test,] # test predictors
  y.test = y[i.test]  # test responses
  
  cv.las.fit = cv.glmnet(X.train, y.train, alpha=1) 

  las.fit = glmnet(X.train, y.train, alpha=1, lambda=cv.las.fit$lambda.min)

  y.train.hat = predict(las.fit, newx = X.train, type = "response")        # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat = predict(las.fit, newx = X.test, type = "response")          # y.test.hat=X.test %*% fit$beta  + fit$a0
  
  res.las.tr[,k] = y.train - y.train.hat
  res.las.ts[,k] = y.test - y.test.hat
  
  Rsq.las.tr[k] = 1 - (  mean((y.train.hat - y.train)^2)  /  mean((y - mean(y))^2)  ) 
  Rsq.las.ts[k] = 1 - (  mean((y.test.hat - y.test)^2)  /  mean((y - mean(y))^2)  )
  
  cat(sprintf("k=%3.f| Rsq.las.ts=%.2f| Rsq.las.tr=%.2f| \n", k,  Rsq.las.ts[k],  Rsq.las.tr[k]))
}


#####################################################################
#Elastic Net Regression
#####################################################################


set.seed(5)


# for loop 100 times to fit model, calculate fit statistics, and store results
for (k in 1:nsamples){
  
  # randomly split data into training and test set
  i.mix = sample(1:n)
  df.split = 1:(n*0.8)
  
  i.train=i.mix[df.split]
  i.test=i.mix[-df.split]
  
  X.train = X[i.train,]   # training predictors
  y.train = y[i.train]    # training responses
  X.test = X[i.test,] # test predictors
  y.test = y[i.test]  # test responses
  
  cv.elnet.fit = cv.glmnet(X.train, y.train, alpha=0.5)

  elnet.fit = glmnet(X.train, y.train, alpha=0.5, lambda=cv.elnet.fit$lambda.min)

  y.train.hat  =     predict(elnet.fit, newx = X.train, type = "response")        # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat   =     predict(elnet.fit, newx = X.test, type = "response")         # y.test.hat=X.test %*% fit$beta  + fit$a0
  
  res.elnet.tr[,k] = y.train - y.train.hat
  res.elnet.ts[,k] = y.test - y.test.hat
  
  Rsq.elnet.tr[k] = 1 - (  mean((y.train.hat - y.train)^2)  /  mean((y - mean(y))^2)  ) 
  Rsq.elnet.ts[k] = 1 - (  mean((y.test.hat - y.test)^2)  /  mean((y - mean(y))^2)  )
  
  cat(sprintf("k=%3.f| Rsq.elnet.ts=%.2f| Rsq.elnet.tr=%.2f| \n", k,  Rsq.elnet.ts[k],  Rsq.elnet.tr[k]))
}


#####################################################################
#Random Forrest 
#####################################################################


set.seed(5)


# for loop 100 times to fit model, calculate fit statistics, and store results
for (k in 1:nsamples){
  
  # randomly split data into training and test set
  i.mix = sample(1:n)
  df.split = 1:(n*0.8)
  
  i.train=i.mix[df.split]
  i.test=i.mix[-df.split]
  
  X.train = X[i.train,]   # training predictors
  y.train = y[i.train]    # training responses
  X.test = X[i.test,] # test predictors
  y.test = y[i.test]  # test responses
  
  # fit RF and calculate and record the train and test R squares 
  rf = randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat = predict(rf, X.test)
  y.train.hat = predict(rf, X.train)

  res.rf.tr[,k] = y.train - y.train.hat
  res.rf.ts[,k] = y.test - y.test.hat
  
  Rsq.rf.tr[k] = 1 - (  mean((y.train.hat - y.train)^2)  /  mean((y - mean(y))^2)  ) 
  Rsq.rf.ts[k] = 1 - (  mean((y.test.hat - y.test)^2)  /  mean((y - mean(y))^2)  )
  
  cat(sprintf("k=%3.f| Rsq.rf.ts=%.2f| Rsq.rf.tr=%.2f| \n", k,  Rsq.rf.ts[k],  Rsq.rf.tr[k]))
}


#####################################################################
# Boxplots of training and test R-squared  
#####################################################################

Rsq.train = matrix(0, nrow = nsamples, ncol = 4)
Rsq.test = matrix(0, nrow = nsamples, ncol = 4)
colnames(Rsq.train) = c("Ridge", "Lasso", "Elastic-Net", "Random Forest")
colnames(Rsq.test) = c("Ridge", "Lasso", "Elastic-Net", "Random Forest")

Rsq.train[,1] = Rsq.rid.tr
Rsq.train[,2] = Rsq.las.tr
Rsq.train[,3] = Rsq.elnet.tr
Rsq.train[,4] = Rsq.rf.tr

Rsq.test[,1] = Rsq.rid.ts
Rsq.test[,2] = Rsq.las.ts
Rsq.test[,3] = Rsq.elnet.ts
Rsq.test[,4] = Rsq.rf.ts


par(mfrow=c(2,1))
boxplot(Rsq.train[,c(1:4)],
        main="Training R-sq For Each Model",
        xlab="Model",
        ylab="R-sq",
        col="orange",
        border="brown"
)
boxplot(Rsq.test[,c(1:4)],
        main="Test R-sq For Each Model",
        xlab="Model",
        ylab="R-sq",
        col="blue",
        border="black"
)

#####################################################################
# 10-fold CV curves for Ridge, Lasso, and Elastic-Net  
#####################################################################

# using 10-fold CV curves from last sample in each model
par(mfrow=c(1,3))
plot(cv.rid.fit, main="Ridge: 10-fold CV", line = 2.5)
plot(cv.las.fit, main="Lasso: 10-fold CV", line = 2.5)
plot(cv.elnet.fit, main="Elastic-Net: 10-fold CV", line = 2.5)


#####################################################################
# Boxplots of training and test residuals  
#####################################################################

# train vs test residuals from last sample in each model
res.train = matrix(0, nrow = n*0.8, ncol = 4)
res.test = matrix(0, nrow = n*0.2, ncol = 4)
colnames(res.train) = c("Ridge", "Lasso", "Elastic-Net", "Random Forest")
colnames(res.test) = c("Ridge", "Lasso", "Elastic-Net", "Random Forest")

res.train[,1] = res.rid.tr[,nsamples] 
res.train[,2] = res.las.tr[,nsamples]
res.train[,3] = res.elnet.tr[,nsamples]
res.train[,4] = res.rf.tr[,nsamples]

res.test[,1] = res.rid.ts[,nsamples]
res.test[,2] = res.las.ts[,nsamples]
res.test[,3] = res.elnet.ts[,nsamples] 
res.test[,4] = res.rf.ts[,nsamples]

par(mfrow=c(2,1))
boxplot(res.train[,c(1:4)],
        main="Training Residuals For Each Model",
        xlab="Model",
        ylab="Residuals",
        col="orange",
        border="brown"
)
boxplot(res.test[,c(1:4)],
        main="Test Residuals For Each Model",
        xlab="Model",
        ylab="Residuals",
        col="blue",
        border="black"
)


#####################################################################
# Bootstrapped Error Bars  
#####################################################################

n = dim(X)[1]
p = dim(X)[2]

bootstrapSamples = 100
beta.rid.bs = matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.las.bs = matrix(0, nrow = p, ncol = bootstrapSamples)         
beta.elnet.bs = matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.rf.bs = matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rid
  a = 0 # ridge
  cv.fit = cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.rid.bs[,m] = as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
  # fit bs las
  a = 1 # lasso
  cv.fit = cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.las.bs[,m] = as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
  # fit bs elnet
  a = 0.5 # elastic-net
  cv.fit = cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit = glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.elnet.bs[,m] = as.vector(fit$beta)
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
  # fit bs rf
  rf = randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m] = as.vector(rf$importance[,1])
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
# calculate bootstrapped standard errors

rid.bs.sd = apply(beta.rid.bs, 1, "sd")
las.bs.sd = apply(beta.las.bs, 1, "sd")
elnet.bs.sd = apply(beta.elnet.bs, 1, "sd")
rf.bs.sd = apply(beta.rf.bs, 1, "sd")


# to view stacked plots with column index:
betaS.rid = data.frame(c(1:p), as.vector(fit$beta), 2*rid.bs.sd)
colnames(betaS.rid) = c( "feature", "value", "err")

betaS.las = data.frame(c(1:p), as.vector(fit$beta), 2*las.bs.sd)
colnames(betaS.las) = c( "feature", "value", "err")

betaS.elnet = data.frame(c(1:p), as.vector(fit$beta), 2*elnet.bs.sd)
colnames(betaS.elnet) = c( "feature", "value", "err")

betaS.rf = data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf) = c( "feature", "value", "err")


# we need to change the order of factor levels by specifying the order explicitly.
betaS.rid$feature = factor(betaS.rid$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.las$feature = factor(betaS.las$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.elnet$feature = factor(betaS.elnet$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
betaS.rf$feature = factor(betaS.rf$feature, levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])


ridPlot = ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  ggtitle("Bootstrap Coefficient Estimates - Ridge") 

lasPlot = ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  ggtitle("Bootstrap Coefficient Estimates - Lasso") 

elnetPlot = ggplot(betaS.elnet, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  ggtitle("Bootstrap Coefficient Estimates - Elastic Net")  

rfPlot = ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  ggtitle("Bootstrap Coefficient Estimates - Random Forest")  


grid.arrange(ridPlot, lasPlot, elnetPlot, rfPlot, nrow = 4)


