
nbm.rmse <- function(y,yhat){
  return(sqrt(mean((y-yhat)^2))/(mean(y)))
}

nbr.rmse <- function(y,yhat){
  return(sqrt(mean((y-yhat)^2))/(max(y)-min(y)))
}
nbiqr.rmse <- function(y,yhat){
  return(sqrt(mean((y-yhat)^2))/(quantile(y,0.75)-quantile(y,0.25)))
}

#96.ahead
MASE.96 = function(y,yhat){
  n = length(y)
  n.1 = length(y)-1
  y.1 = y[-c(1:96)]
  y.2 = y[-c(((n-96+1):n))]
  q = (y-yhat)/(1/(n-96)*sum(abs(y.1-y.2)))
  return(mean(abs(q)))
}

mae <- function(y,yhat){
  return(mean(abs(y-yhat)))
}

mse <- function(y,yhat){
  return(mean((y-yhat)^2))
}
