## install and load packages
libraries = c("softImpute")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

## Cross-validation for the tuning parameter
cv.lambda    = function(lamb, X, Y, tau) {
  K        = 5
  LOSS     = 0
  groupnum = floor(dim(X)[1]/K)
  r.idx    = sample(1:dim(X)[1], dim(X)[1], replace=F)
  for (k in 1:K) {
    vali_idx = r.idx[(groupnum * k - groupnum + 1):(groupnum * k)]
    vali_Y   = Y[vali_idx, ]
    vali_X   = X[vali_idx, ]
    train_X  = X[-vali_idx, ]
    train_Y  = Y[-vali_idx, ]
    n        = dim(vali_Y)[1]
    m        = dim(vali_Y)[2]
    fit      = mer.als(Y = train_Y, X = train_X, tau = tau, epsilon = 1e-06, lambda = lamb, 
                       itt  = 2000,RK = floor(ncol(X)/2))
    loss     = sum(abs(tau - matrix(as.numeric(vali_Y - vali_X %*% fit$Gamma < 0), n, m)) 
                   * (vali_Y - vali_X %*% fit$Gamma)^2)
    LOSS     = c(LOSS, loss)
  }
  sum(LOSS)
}