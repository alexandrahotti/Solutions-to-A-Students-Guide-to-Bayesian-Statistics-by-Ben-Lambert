import numpy as np
from scipy.stats import gamma, poisson


class Disease_outbreaks_model:

    def __init__(self, a, b, data):
        self.a = a
        self.b = b
        self.data = data

    def gamma_prior(self):

        lambda_lower = 0
        lambda_upper = 20

        lambda_range = np.linspace(lambda_lower, lambda_upper, 100)

        gamma_prior = gamma.pdf(lambda_range, a=self.a,
                                loc=0, scale=(1 / self.b))

        return gamma_prior, lambda_range

    def gamma_posterior(self):
        """ Using at Poission likelihood and a Gamma prior gives us a conjugate
        Gamma posterior. """

        lambda_lower = 0
        lambda_upper = 20

        lambda_range = np.linspace(lambda_lower, lambda_upper, 100)

        a_prim = self.a + np.sum(self.data)
        b_prim = self.b + len(self.data)
        posterior = gamma.pdf(lambda_range, a=a_prim,
                                loc=0, scale=(1 / b_prim))

        return posterior, lambda_range

    def sample_posterior_predictive_distribution(self, no_iterations):
        """ The posterior predictive sampling using a Gamma(3 + 35, 0.5 + 5)
        posterior and a Poisson(lambda) likelihood. """


        posterior_predictive_disease_outbreaks = []

        a_prim = self.a + np.sum(self.data)
        b_prim = self.b + len(self.data)

        for i in range(no_iterations):

            """ Sampling a lambda value from the posterior. """
            sample_sz =  1
            lambda_i = gamma.rvs(a_prim, loc=0, scale=(1/b_prim), size=sample_sz)

            """ Sampling x_i (no disease outbreaks a certain year) for the new data from a Poisson(lambda_i) Likelihood for this new data over lambda_i. """
            sample_sz =  1
            disease_outbreaks_i = poisson.rvs(lambda_i, loc=0, size=sample_sz)

            posterior_predictive_disease_outbreaks.append(disease_outbreaks_i[0])

        return posterior_predictive_disease_outbreaks



    # def posterior_predictive(self, a, b, sample_size_new_data, no_iterations):
    #     """ The posterior predictive sampling using a Gamma(3 + 35, 0.5 + 5)
    #     posterior and a Poisson(lambda) likelihood. """
    #
    #     posterior_predictive_disease_occurances = []
    #
    #     for i in range(no_iterations):
    #         """ Sampling a theta value from the posterior. """
    #         sample_sz =  1
    #         theta_i = np.random.beta(self.no_disease_occurances + a, self.sample_size + b - self.no_disease_occurances, sample_sz)
    #
    #         """ Sampling x_i (disease occurances) for the new data from a Binomial(100, theta_i) Likelihood for this new data over theta_i. """
    #         sample_sz =  1
    #         disease_occurance_sample = binom.rvs(sample_size_new_data, theta_i, 0, sample_sz)
    #
    #         posterior_predictive_disease_occurances.append(disease_occurance_sample[0])
    #
    #     return posterior_predictive_disease_occurances
