## Function: Computing the gradient of the expectile loss function
G.er = function(A, Y, X, tau, m, n) {
  W          = (Y - X %*% A)
  index_p    = which(W > 0, arr.ind = TRUE)
  W[index_p] = 2 * tau
  index_n    = which(W < 0, arr.ind = TRUE)
  W[index_n] = 2 - 2 * tau
  W          = W * (Y - X %*% A)
  temp       = (-t(X) %*% W)/(m * n)
  temp
}