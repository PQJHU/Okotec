## install and load packages
libraries = c("softImpute")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

## Function: FISTA algorithm
mer.als = function(Y, X, tau, lambda, epsilon = 10^(-6), itt = 2000, RK = floor(ncol(X)/2)) {
  ## Initialize
  m = ncol(Y)
  n = nrow(Y)
  p = ncol(X)
  L       = 2 * (m * n)^(-1) * max(tau, (1 - tau)) * norm(X, type = "F")^2
  Omega   = matrix(0, nrow = p, ncol = m)
  delta   = 1  # step size
  error   = 1e+07
  L.error = 1e+10
  it      = 1
  ## Output
  A       = matrix(0, nrow = p, ncol = m)
  A_      = matrix(0, nrow = p, ncol = m)
  ## Main iteration
  while (it < itt & error > epsilon) {
    S         = svd.als(Omega - L^(-1) * G.er(Omega, Y, X, tau, m = m, n = n), rank.max = RK)
    temp.sv   = S$d - (lambda/L)
    temp.sv[temp.sv < 0] = 0
    A         = S$u %*% diag(temp.sv) %*% t(S$v)
    delta.for = (1 + sqrt(1 + 4 * delta^2))/2
    Omega     = A + (delta - 1)/(delta.for) * (A - A_)
    error     = L.error - (m * n)^(-1) * (sum(abs(tau - matrix(as.numeric(Y - X %*% A < 0), n, m)) * 
                                                (Y - X %*% A)^2) + lambda * sum(temp.sv))
    L.error   = (m * n)^(-1) * sum(abs(tau - matrix(as.numeric(Y - X %*% A < 0), n, m)) * (Y - X %*% 
                                                                                             A)^2) + lambda * sum(temp.sv)
    A_        = A
    delta     = delta.for
    it        = it + 1
    print(c(error, delta, (m * n)^(-1) * sum(abs(tau - matrix(as.numeric(Y - X %*% A < 0), n, m)) * 
                                               (Y - X %*% A)^2), sum(temp.sv)))
    # if(it < 10){error=1000000}
  }
  list(Gamma = A, d = S$d, U = S$u, V = S$v, error = error, loss = (m * n)^(-1) * sum(abs(tau - matrix(as.numeric(Y - 
                                                                                                                    X %*% A < 0), n, m)) * (Y - X %*% A)^2), norm = sum(temp.sv), lambda = lambda, 
       iteration = it)
}