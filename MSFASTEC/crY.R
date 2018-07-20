cr.Y <- function(W){
  id.gr <- which(y.hat==exp.a[W])
  Y.s1 <- Y.train[id.gr]
  YYY.1 <- c()
  seq.1 <- seq(1,floor(length(Y.s1)/96)*96, by=96)
  seq.2 <- seq(1,floor(length(Y.s1)/96)*96+1, by=96)[-1]
  for(i in 1:floor(length(Y.s1)/96)){
    YYY.1 <- cbind(YYY.1,Y.s1[seq.1[i]:(seq.2[i]-1)])
  }
  set.seed(1)
  y.samp <- sample(1:floor(length(Y.s1)/96),240,replace=T)
  
  Y.b1 <- YYY.1[,y.samp]
  return(Y.b1)
}