import numpy as np
from scipy.stats import gamma
from scipy.stats import nbinom


class Epilepsy_model:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def posterior_gamma(self, epilepsy_counts):

        theta_lower = 0
        theta_upper = 20

        theta_range = np.linspace(theta_lower, theta_upper, 1000)

        a_prim = np.sum(epilepsy_counts) + self.a
        b_prim = self.b + len(epilepsy_counts)

        gamma_posterior = gamma.pdf(theta_range, a = a_prim, scale = (1 /b_prim))

        return gamma_posterior, theta_range


    def negativebinomial_pmf(self, x, mu, kappa):

        n = kappa
        p = float(kappa) / (kappa + mu)

        negbinomial_pdf = nbinom.pmf(x, n, p)

        return negbinomial_pdf


    def posterior_predictive_gamma(self, epilepsy_counts):

        theta_lower = 0
        theta_upper = 20

        theta_range = np.linspace(theta_lower, theta_upper, 100)

        kappa = len(epilepsy_counts)*np.mean(epilepsy_counts) + self.a
        p = (self.b + len(epilepsy_counts)) #/(self.b + len(epilepsy_counts)+1)

        print(kappa)
        print(p)

        gamma_posterior = self.negativebinomial_pmf(theta_range, np.sum(epilepsy_counts)+ self.a, 1/kappa)


        return gamma_posterior, theta_range
