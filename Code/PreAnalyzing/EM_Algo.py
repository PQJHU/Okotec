import numpy as np
import sklearn
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import norm


def uni_normal(x, mu, sigma):
    return list(map(lambda mu, sigma:
                    (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp((-(x - mu) ** 2) / (2 * (sigma ** 2))),
                    mu, sigma))


# mu_init = 63800
# sigma_init = 22500


class EM_Algo_1D:

    def __init__(self, X, K, mu_init, sigma_init, stop_criteria):
        self.X = X
        self.K = K
        self.weight = [1 / K] * K
        self.mu = mu_init  # list with K params
        self.sigma = sigma_init  # list with K params
        # self.gamma_sum = self.mu_numerator = self.sigma_numerator = [0.5] * K
        self.X, self.weight, self.mu, self.sigma = map(np.asarray, (self.X, self.weight, self.mu, self.sigma))
        self.stop = stop_criteria

        # self.sample_count = 0

    def E_Step(self, x):
        """
        Given an observation x, calculate the responding coefficient of related to each model gamma
        :param x: single sample
        :return: Expectation for one sample
        """
        pdf_weighted = self.weight * norm.pdf(x, self.mu, self.sigma)
        self.expectation = pdf_weighted / pdf_weighted.sum()
        # print(f'pdf_weight:{pdf_weighted}')
        # print(f'expectation:{self.expectation}')
        # print(type(self.expectation))

    def M_Step(self):
        """
        Update parameters, weight, mu, sigma
        :return: Updated parameters to instance
        """

        expectation_sum = np.asarray([0.] * self.K)
        sigma_numerator = np.asarray([0.] * self.K)
        mu_numerator = np.asarray([0.] * self.K)
        # print(self.X)
        # print(type(self.X))
        N = len(self.X)

        for i, x in enumerate(self.X):
            """
            Calculate the aggregation of N samples
            """
            # print(f'Sample {i}: {x}')
            # Run Expectation on sample i for K models
            self.E_Step(x=x)
            expectation_sum += self.expectation
            # print(f'Expectation Sum:{expectation_sum}')
            sigma_numerator += self.expectation * ((x - self.mu) ** 2)
            mu_numerator += self.expectation * x

        # Update weight
        self.weight = expectation_sum / N
        print(f'weight: {self.weight}')

        # Update sigma
        self.sigma = np.sqrt(sigma_numerator / expectation_sum)
        print(f'sigma:{self.sigma}')

        # Update mu
        mu_old = self.mu
        self.mu = mu_numerator / expectation_sum
        print(f'mu:{self.mu}')

        # Update length
        self.epsilon = max((abs(np.subtract(self.mu, mu_old))))
        print(f'Epsilon: {self.epsilon}')

    def fit(self, iterate_num):

        for iterate in range(iterate_num):
            print(f'Iteration:{iterate}')
            self.M_Step()
            if self.epsilon < self.stop:
                print('*************Converged**********')
                break


# ======================Algorithm test===================
# Generate 6 gaussian distr. mixture data to test EM

# np.random.seed(666)
# mu = [0, 4, -10, 10]
# sigma = [1, 0.5, 5, 3]
# weight = [0.3, 0.1, 0.4, 0.2]
# N = 10000
# cluster_num = 4
#
#
# def generate_GaussianMixture(weight, mu, sigma, sample_num):
#     sample_sizes = np.multiply(weight, sample_num)
#     full_sample = list()
#     params = list(zip(sample_sizes, mu, sigma))
#
#     for param in params:
#         print(param)
#         sample = np.random.normal(loc=param[1], scale=param[2], size=int(param[0]))
#         full_sample.extend(sample.tolist())
#     return full_sample
#
#
# test_set = generate_GaussianMixture(weight, mu, sigma, N)
#
# ran_idx = np.random.choice(len(test_set), cluster_num)
# mu_init = [test_set[i] for i in ran_idx]
# sigma_init = np.random.randint(1, 10, cluster_num).tolist()
#
# test_em = EM_Algo_1D(X=test_set, K=cluster_num, mu_init=mu_init, sigma_init=sigma_init, stop_criteria=0.001)
# test_em.fit(iterate_num=5000)

"""
Iteration:164
weight: [0.35671034 0.02090645 0.37687762 0.24550558]
sigma:[4.90357395 2.09474294 4.85718307 0.85108107]
mu:[-10.56295461  -9.1044208    6.23992808  -0.12910715]
Epsilon: 0.0009835600009377998
*************Converged**********
"""
