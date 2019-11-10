from helper_functions import plotter, plotter_histogram
from Disease_outbreaks_model import *
import numpy as np
# import matplotlib.pyplot as plt



def main():

    # data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\conjugate_epil.csv')
    # epilepsy_data = loadData(data_path)
    #
    # epilepsy_counts = get_column(epilepsy_data["x"])

    # a = 3
    # b = 0.5
    # data = [3, 7, 4, 10, 11]
    #
    # disease_outbreaks_model = Disease_outbreaks_model(a, b, data)

    """ Q 10.1.1 The Gamma Prior """
    # gamma_prior, lambda_range = disease_outbreaks_model.gamma_prior()
    # plotter(lambda_range, gamma_prior, 'The prior '+ r'$\Gamma$' +'(3, 0.5)', r'$\lambda$', "pdf")

    """ Q 10.1.2 The Gamma conjugate Prior of the Poisson Likelihood"""
    # gamma_posterior, lambda_range = disease_outbreaks_model.gamma_prior()
    # plotter(lambda_range, gamma_posterior, 'The posterior '+ r'$\Gamma$' +'(3 + 35, 0.5 + 5)', r'$\lambda$', "pdf")


    """ Q 10.1.3-4 Posterior predictive"""
    # no_iterations = 10000
    # posterior_predictive_disease_outbreaks = disease_outbreaks_model.sample_posterior_predictive_distribution(no_iterations)
    #
    # # plotter_histogram(posterior_predictive_disease_outbreaks, 'No disease outbreaks', 'Frequency', 'Posterior Predictive distribution over disease outbreaks')
    #
    #
    # print("Data maximum: ", np.max(data))
    # print("Data minimum: ", np.min(data))
    #
    # print('Pr(T(fake) >= T(actual)_max | data) ', np.mean(np.array(posterior_predictive_disease_outbreaks) >= np.max(data)))
    # print('Pr(T(fake) <= T(actual)_min | data) ', np.mean(np.array(posterior_predictive_disease_outbreaks) <= np.min(data)))
    #
    #
    # """ Q 10.1.6 Posterior predictive"""
    # print('Pr(T(fake) >= 20 | data) ', np.mean(np.array(posterior_predictive_disease_outbreaks) >= 20 ))


    """ Q 10.1.7-8 Posterior predictive"""

    a = 3
    b = 0.5
    data = [3, 7, 4, 10, 11, 20]

    disease_outbreaks_model = Disease_outbreaks_model(a, b, data)

    # New poserior
    gamma_posterior, lambda_range = disease_outbreaks_model.gamma_prior()
    plotter(lambda_range, gamma_posterior, 'The posterior '+ r'$\Gamma$' +'(3 + 35 + 20, 0.5 + 5 + 1)', r'$\lambda$', "pdf")

    no_iterations = 10000
    posterior_predictive_disease_outbreaks = disease_outbreaks_model.sample_posterior_predictive_distribution(no_iterations)
    print('Pr(T(fake) >= 20 | data) ', np.mean(np.array(posterior_predictive_disease_outbreaks) >= 20 ))
    
if __name__ == '__main__':
    main()
