# clear variables and close windows
rm(list = ls(all = TRUE))
graphics.off()

############################################################################
######################### INSTALL RELEVANT PACKAGES ########################
############################################################################
setwd("~/Desktop/MSFASTEC")

libraries = c("splines", "matrixStats", "softImpute","expectreg","zoo")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

source("mer.als.R")
source("ger.R")
source("cvlossals.R")
source("yfind.R")
source("accuracy.R")
source("crY.R")
########################################################
######################### START ########################
########################################################

data <- read.csv("last_anonym_2017_vartime.csv",header=T,sep=";")

y <- data[,2]
y.t <- y/(max(y)+1e3)
Y   <- log(y.t/(1-y.t))
y.tt <- y.t[(200*96+1):(365*96)]
Y.train <-  Y[1:(200*96)]
Y.test <- Y[(200*96+1):(365*96)]
X.1 <- data[,c(3:20)]
X.1.01 <-  X.1/matrix(apply(X.1,2,max),nrow(X.1),ncol(X.1),byrow=T)
X.2 <- data[,21:44]
X.2.01 <-  X.2/matrix(apply(X.2,2,max),nrow(X.2),ncol(X.2),byrow=T)
X.3 <- data[,c(45:74)]
X.3.01 <-  X.3/matrix(apply(X.3,2,max),nrow(X.3),ncol(X.3),byrow=T)

XX <- cbind(rowSums(X.1.01),rowSums(X.2.01),rowSums(X.3.01))
colnames(XX) <- c("G1","G2","G3")
y.trash <- ifelse(y<50000,1,0)


################################################################
######################### IN-SAMPLE FIT ########################
############## BASED ON LEVELS & SMALLEST DEVIANCE #############
################################################################

exp.a <- expectile(Y,p=c(0.0001,0.001, 0.01,0.03,seq(0.1,0.97,length=10),0.993,0.999))
plot(Y, type="l")
abline(h=exp.a)
mu.hat <- exp.a
y.hat <- c()
for(i in 1:length(Y.train)){
  print(Y.train[i])
  t.stat <- abs(Y.train[i]-mu.hat)/min(sqrt((Y.train[i]-mu.hat)^2))-1
  y.hat[i] <- mu.hat[which(t.stat==min(t.stat))]
  print(y.hat[i])
}
lines(y.hat,col="red")
plot((1+exp(-Y.train))^(-1),type="l",ylab="",xlab="")
lines((1+exp(-y.hat))^(-1),col="red")

mae((1+exp(-Y.train))^(-1),(1+exp(-y.hat))^(-1))
# 0.01038724

#########################################################################
#########################  GET RELEVANT PERIODS  ########################
#########################################################################

mu.hat <- exp.a
time.d <- zooreg(1:(365),start=as.Date("2017-01-01"))
date   <-  rep(time(zooreg(1:(365),start=as.Date("2017-01-01"))),each=96)

dates <- cbind(substr(date,1,4),substr(date,6,7),substr(date,9,10))

if.use=c()
for(i in 1:length(exp.a)){if.use[i] <- floor(length(which(y.hat==exp.a[i]))/96)}
use.only <- which(if.use>3)

Y.b1 <- lapply(X=use.only, function(X){cr.Y(X)})

############################################################################
#########################  SET RELEVANT PARAMETERS  ########################
############################################################################

n    <- 96
p    <- ceiling(n^0.6)
TAU  <- c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,0.999)

############################################################################
#########################  SET-UP  B-SPLINES FOR Y  ########################
############################################################################

xx       <- seq(0, 1, length = n)
X.fac    <- bs(xx, df = p, intercept = TRUE)
X.v      <- matrix(0, nrow = n, ncol = 0)
for (i in 1:p) {
  X.v  <- cbind(X.v, X.fac[, i])
}

############################################################################
#########################    FASTEC   ALS   MODEL   ########################
############################################################################
for(lo in 1:length(Y.b1)){
  y.m <- Y.b1[[lo]]
  k        <- 0
  Y1       <- data.matrix(y.m)
  m        <- dim(Y1)[2]
  sig_x    <- sqrt(norm(X.v, type = "F")/n)
  lamb.1 <- optimize(cv.lambda, c(1e-05, 0.01), X = X.v, Y = Y1, tau = 0.5)$minimum 
  save(lamb.1, file=paste("lamb",lo,".RData",sep=""))
  fit.1  <- lapply(X = c(1:length(TAU)), FUN = function(X){mer.als(Y = Y1, X = X.v, tau = TAU[X], epsilon = 1e-06, lambda = lamb.1, itt = 2000)})
  save(fit.1, file=paste("fit",lo,".RData",sep=""))
}
save(X.v, file="Bsplines.RData")
# load("Bsplines.RData")

############################################################################
#############################  RETRIEVE FACTORS  ###########################
############################################################################
TAU       <- c(0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,0.999)
for(lo in 1:length(Y.b1)){
  load("fit1.RData") 
  fit.t <- fit.1
  assign(paste0("fit.", lo), fit.t)
}

econU.DK1 <- lapply(fit.1, "[[", "U")
econd.DK1 <- lapply(lapply(fit.1, "[[", "d"), diag)
econV.DK1 <- lapply(lapply(fit.1, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK1 <- lapply(econV.DK1, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK2 <- lapply(fit.2, "[[", "U")
econd.DK2 <- lapply(lapply(fit.2, "[[", "d"), diag)
econV.DK2 <- lapply(lapply(fit.2, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK2 <- lapply(econV.DK2, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK3 <- lapply(fit.3, "[[", "U")
econd.DK3 <- lapply(lapply(fit.3, "[[", "d"), diag)
econV.DK3 <- lapply(lapply(fit.3, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK3 <- lapply(econV.DK3, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK4 <- lapply(fit.4, "[[", "U")
econd.DK4 <- lapply(lapply(fit.4, "[[", "d"), diag)
econV.DK4 <- lapply(lapply(fit.4, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK4 <- lapply(econV.DK4, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK5 <- lapply(fit.5, "[[", "U")
econd.DK5 <- lapply(lapply(fit.5, "[[", "d"), diag)
econV.DK5 <- lapply(lapply(fit.5, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK5 <- lapply(econV.DK5, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK6 <- lapply(fit.6, "[[", "U")
econd.DK6 <- lapply(lapply(fit.6, "[[", "d"), diag)
econV.DK6 <- lapply(lapply(fit.6, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK6 <- lapply(econV.DK6, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK7 <- lapply(fit.7, "[[", "U")
econd.DK7 <- lapply(lapply(fit.7, "[[", "d"), diag)
econV.DK7 <- lapply(lapply(fit.7, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK7 <- lapply(econV.DK7, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK8 <- lapply(fit.8, "[[", "U")
econd.DK8 <- lapply(lapply(fit.8, "[[", "d"), diag)
econV.DK8 <- lapply(lapply(fit.8, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK8 <- lapply(econV.DK8, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK9 <- lapply(fit.9, "[[", "U")
econd.DK9 <- lapply(lapply(fit.9, "[[", "d"), diag)
econV.DK9 <- lapply(lapply(fit.9, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK9 <- lapply(econV.DK9, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))

econU.DK10 <- lapply(fit.10, "[[", "U")
econd.DK10 <- lapply(lapply(fit.10, "[[", "d"), diag)
econV.DK10 <- lapply(lapply(fit.10, "[[", "V"), "[", TRUE, c(1:floor(ncol(X.v)/2)))
econV.DK10 <- lapply(econV.DK10, "colnames<-", paste0("load", seq(1, floor(ncol(X.v)/2))))


library(vars); library(rmgarch)
############################################################################
#############################   VAR(p) FORECAST    #########################
#############################    OUT OF SAMPLE     #########################
############################################################################

p.var.DK11      <- c()
var.fit.DK1    <- list()
var.fcst.DK1   <- list()
fcst.curve.DK1 <- list()
p.var.DK22      <- c()
var.fit.DK2    <- list()
var.fcst.DK2   <- list()
fcst.curve.DK2 <- list()
p.var.DK33      <- c()
var.fit.DK3    <- list()
var.fcst.DK3   <- list()
fcst.curve.DK3 <- list()
p.var.DK44      <- c()
var.fit.DK4    <- list()
var.fcst.DK4   <- list()
fcst.curve.DK4 <- list()
p.var.DK55      <- c()
var.fit.DK5    <- list()
var.fcst.DK5   <- list()
fcst.curve.DK5 <- list()
p.var.DK66      <- c()
var.fit.DK6    <- list()
var.fcst.DK6   <- list()
fcst.curve.DK6 <- list()
p.var.DK77      <- c()
var.fit.DK7    <- list()
var.fcst.DK7   <- list()
fcst.curve.DK7 <- list()
p.var.DK88      <- c()
var.fit.DK8    <- list()
var.fcst.DK8   <- list()
fcst.curve.DK8 <- list()
p.var.DK99      <- c()
var.fit.DK9    <- list()
var.fcst.DK9   <- list()
fcst.curve.DK9 <- list()
p.var.DK1100      <- c()
var.fit.DK10    <- list()
var.fcst.DK10   <- list()
fcst.curve.DK10 <- list()

for(i in 1:length(TAU)){
  p.var.DK1           <- 1
  var.fit.DK1[[i]]    <- varxfit((econV.DK1[[i]][1:(200),]), p=p.var.DK1, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK1[[i]]   <- varxfilter((econV.DK1[[i]][(200+1):(240),]), p=p.var.DK1, Bcoef=(var.fit.DK1[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK1[[i]] <- (X.v %*% econU.DK1[[i]] %*% econd.DK1[[i]] %*% (t(var.fcst.DK1[[i]]))) # create curve
  p.var.DK11[i]       <- p.var.DK1
  p.var.DK2           <- 1
  var.fit.DK2[[i]]    <- varxfit((econV.DK2[[i]][1:(200),]), p=p.var.DK2, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK2[[i]]   <- varxfilter((econV.DK2[[i]][(200+1):(240),]), p=p.var.DK2, Bcoef=(var.fit.DK2[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK2[[i]] <- (X.v %*% econU.DK2[[i]] %*% econd.DK2[[i]] %*% (t(var.fcst.DK2[[i]]))) # create curve
  p.var.DK22[i]       <- p.var.DK2
  p.var.DK3           <- 1
  var.fit.DK3[[i]]    <- varxfit((econV.DK3[[i]][1:(200),]), p=p.var.DK3, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK3[[i]]   <- varxfilter((econV.DK3[[i]][(200+1):(240),]), p=p.var.DK3, Bcoef=(var.fit.DK3[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK3[[i]] <- (X.v %*% econU.DK3[[i]] %*% econd.DK3[[i]] %*% (t(var.fcst.DK3[[i]]))) # create curve
  p.var.DK33[i]       <- p.var.DK3
  p.var.DK4           <- 1
  var.fit.DK4[[i]]    <- varxfit((econV.DK4[[i]][1:(200),]), p=p.var.DK4, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK4[[i]]   <- varxfilter((econV.DK4[[i]][(200+1):(240),]), p=p.var.DK4, Bcoef=(var.fit.DK4[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK4[[i]] <- (X.v %*% econU.DK4[[i]] %*% econd.DK4[[i]] %*% (t(var.fcst.DK4[[i]]))) # create curve
  p.var.DK44[i]       <- p.var.DK4
  p.var.DK5           <- 1
  var.fit.DK5[[i]]    <- varxfit((econV.DK5[[i]][1:(200),]), p=p.var.DK5, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK5[[i]]   <- varxfilter((econV.DK5[[i]][(200+1):(240),]), p=p.var.DK5, Bcoef=(var.fit.DK5[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK5[[i]] <- (X.v %*% econU.DK5[[i]] %*% econd.DK5[[i]] %*% (t(var.fcst.DK5[[i]]))) # create curve
  p.var.DK55[i]       <- p.var.DK5
  p.var.DK6           <- 1
  var.fit.DK6[[i]]    <- varxfit((econV.DK6[[i]][1:(200),]), p=p.var.DK6, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK6[[i]]   <- varxfilter((econV.DK6[[i]][(200+1):(240),]), p=p.var.DK6, Bcoef=(var.fit.DK6[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK6[[i]] <- (X.v %*% econU.DK6[[i]] %*% econd.DK6[[i]] %*% (t(var.fcst.DK6[[i]]))) # create curve
  p.var.DK66[i]       <- p.var.DK6
  p.var.DK7           <- 1
  var.fit.DK7[[i]]    <- varxfit((econV.DK7[[i]][1:(200),]), p=p.var.DK7, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK7[[i]]   <- varxfilter((econV.DK7[[i]][(200+1):(240),]), p=p.var.DK7, Bcoef=(var.fit.DK7[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK7[[i]] <- (X.v %*% econU.DK7[[i]] %*% econd.DK1[[i]] %*% (t(var.fcst.DK7[[i]]))) # create curve
  p.var.DK77[i]       <- p.var.DK7
  p.var.DK8           <- 1
  var.fit.DK8[[i]]    <- varxfit((econV.DK8[[i]][1:(200),]), p=p.var.DK1, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK8[[i]]   <- varxfilter((econV.DK8[[i]][(200+1):(240),]), p=p.var.DK8, Bcoef=(var.fit.DK8[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK8[[i]] <- (X.v %*% econU.DK8[[i]] %*% econd.DK8[[i]] %*% (t(var.fcst.DK8[[i]]))) # create curve
  p.var.DK88[i]       <- p.var.DK8
  p.var.DK9           <- 1
  var.fit.DK9[[i]]    <- varxfit((econV.DK9[[i]][1:(200),]), p=p.var.DK9, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK9[[i]]   <- varxfilter((econV.DK9[[i]][(200+1):(240),]), p=p.var.DK9, Bcoef=(var.fit.DK9[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK9[[i]] <- (X.v %*% econU.DK9[[i]] %*% econd.DK9[[i]] %*% (t(var.fcst.DK9[[i]]))) # create curve
  p.var.DK99[i]       <- p.var.DK9
  p.var.DK10           <- 1
  var.fit.DK10[[i]]    <- varxfit((econV.DK10[[i]][1:(200),]), p=p.var.DK10, constant=F)#, exogen=exo.1)                                         # in-sample fit
  var.fcst.DK10[[i]]   <- varxfilter((econV.DK10[[i]][(200+1):(240),]), p=p.var.DK10, Bcoef=(var.fit.DK10[[i]]$Bcoef))$xfitted#, exogen=exo.2)$xfitted  # out-of-sample backtesting
  fcst.curve.DK10[[i]] <- (X.v %*% econU.DK10[[i]] %*% econd.DK10[[i]] %*% (t(var.fcst.DK10[[i]]))) # create curve
  p.var.DK1100[i]       <- p.var.DK10
}

############################################################################
#############################   OPTIMAL WEIGHTS    #########################
###############################    IN-SAMPLE     ###########################
############################################################################


# optimal parameters
b <- nlm(y.find,c(rep(0.5,length(exp.a)+4)))$estimate

############################################################################
#############################         Y_hat        #########################
#############################    OUT-OF-SAMPLE     #########################
############################################################################

Y.fcst.96 <- c()

train =(1):(200*96)
test = (200*96+1):(365*96)
wdw.seq.1 <- seq(1,(365*96),by=96)-(200*96)
wdw.seq.2 <- seq(96,(365*96+1),by=96)-(200*96)

for(i in 201:364){
  wdw.1 <- (1+((i-30)-1)*96):((i-1)*96) 
  wdw.2 <- ((i-1)*96+1-9):((i)*96-9)
  mu.hat <-  exp.a
  ex.1 <- XX[wdw.2,1]
  ex.2 <- XX[wdw.2,2]
  ex.3 <- XX[wdw.2,3]
  ex.4 <- y.trash[wdw.2+9]
  j   <- ((i-1)*96)
  k.1 <- j+1
  k.2 <- j+96
  print(paste(i, "=", round(Y[j],3),round(Y[k.1],3),round(Y[k.2],3)))
  t.stat <- which(((abs(mean(Y[wdw.2])-mu.hat))/min(sqrt((mean(Y[wdw.2])-mu.hat)^2))-1)==0)
  Y.fcst.96[wdw.seq.1[i]:wdw.seq.2[i]] <- if(t.stat==1){
    b[1]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==2){
    b[2]*fcst.curve.DK1[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==3){
    b[3]*fcst.curve.DK2[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==4){
    b[4]*fcst.curve.DK3[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==5){
    b[5]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==6){
    b[6]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==7){
    b[7]*fcst.curve.DK4[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==8){
    b[8]*mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==9){
    b[9]*fcst.curve.DK5[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==10){
    b[10]* fcst.curve.DK6[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==11){
    b[11]*fcst.curve.DK7[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==12){
    b[12]* fcst.curve.DK8[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==13){
    b[13]* fcst.curve.DK9[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==14){
    b[14]* fcst.curve.DK10[[7]][1:96,1]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==15){
    b[15]* mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }else if(t.stat==16){
    b[16]* mu.hat[t.stat]+b[17]*ex.1+b[18]*ex.2+b[19]*ex.3+b[20]*ex.4
  }
  print(paste(i, "=", round(Y[j],3),round(Y[k.1],3),round(Y[k.2],3),Y.fcst.96[wdw.seq.1[i]],Y.fcst.96[wdw.seq.2[i]]))
}

###################################################
########### Plot before back-transform ############
###################################################
plot(Y.test[(1):(length(Y.fcst.96))],type="l")
lines(Y.fcst.96[1:(length(Y.fcst.96))],col="red")

#######################################
########### back-transform ############
#######################################

Y.tr <- (1+exp(-Y.test[(1):(length(Y.fcst.96))]))^(-1)
Y.hat <- (1+exp(-Y.fcst.96[(1):(length(Y.fcst.96))]))^(-1)
time.d <- zooreg(1:(365),start=as.Date("2017-01-01"))
date   <-  rep(time(zooreg(1:(365),start=as.Date("2017-01-01"))),each=96)
date.1 <- (1:length(Y.train))/(365*96) + (2017) 
date.2 <- (1:length(Y.fcst.96))/(365*96) + (date.1[length(date.1)])

plot(date.2,Y.tr,type="l",ylim=c(0,1), ylab="",xlab="")
title("MS-based FASTEC")
lines(date.2,Y.hat,col="red")

############################################
######### Out of sample performance ########
############################################

# normalised RMSE by range
round(nbr.rmse(Y.tr,Y.hat),4)
# 0.0866
# normalised RMSE by mean
round(nbm.rmse(Y.tr,Y.hat),4)
# 0.1074
# normalised RMSE by IQR
round(nbiqr.rmse(Y.tr,Y.hat),4)
# 0.791 
# Relative to 96 observation before
round(MASE.96(Y.tr,Y.hat),4)
# 0.3388
round(mae(Y.tr,Y.hat),4)
# 0.0454


