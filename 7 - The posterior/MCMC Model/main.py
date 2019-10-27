import pandas as pd
from helper_functions import plotter
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import pymc3

import statsmodels.api as sm


def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    return data_frame


def get_column(data_column):

    data_values = data_column.values.reshape(-1)

    return data_values


def remove_missing_data(gdp_data, mortality_data):

    gdps = []
    mortalities = []
    for i in range(len(gdp_data)):
        if str(gdp_data[i]) != "nan":
            if str(mortality_data[i]) != "nan":
                gdps.append(gdp_data[i])
                mortalities.append(mortality_data[i])

    return gdps, mortalities


def fit_normal_dist_model(gdps, mortalities):

    regressor = LinearRegression()

    regressor.fit(np.log(np.array(gdps).reshape(-1, 1)),
                  np.log(np.array(mortalities).reshape(-1, 1)))  # training the algorithm

    # To retrieve the intercept:
    intercept = regressor.intercept_[0]
    print("Regression intercept ", intercept)

    # For retrieving the slope:
    slope = regressor.coef_[0][0]
    print("Regression coeficient ", slope)

    gdp_log_lower = 2.5
    gdp_log_upper = 11

    gdp_log_range = np.arange(gdp_log_lower, gdp_log_upper, 0.5)
    mortality_log_values = [gdp * slope + intercept for gdp in gdp_log_range]

    return gdp_log_range, mortality_log_values, intercept, slope


def calculate_standard_deviation(values):

    standard_error = np.std(values)

    return standard_error


def calculate_degrees_of_freedom(values):

    n = len(values)
    dof = n - 1

    return dof


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\posterior_posteriorsGdpInfantMortality.csv')
    GDP_data = loadData(data_path)

    alphas = get_column(GDP_data["alpha"])
    betas = get_column(GDP_data["beta"])
    sigmas = get_column(GDP_data["sigma"])

    dof = calculate_degrees_of_freedom(alphas)

    std_alpha = calculate_standard_deviation(alphas)
    std_beta = calculate_standard_deviation(betas)
    std_sigma = calculate_standard_deviation(sigmas)

    print("No. Degrees of Freedom: ", dof)
    print()
    print("Standard Deviations ")
    print("alpha: ", std_alpha)
    print("beta: ", std_beta)
    print("sigma: ", std_sigma)

    print()
    """ Credible intervals of 80 % around the highest density regions. """
    print("Credible intervals of 80 % around the highest density regions")
    print("alpha: ", pymc3.stats.hpd(alphas, alpha=0.2))
    print("beta: ", pymc3.stats.hpd(betas, alpha=0.2))
    print("sigma: ", pymc3.stats.hpd(sigmas, alpha=0.2))

    print()
    """ For comparison of the frequentist sigma value a point estimate of the
    sigma value is calculated. """
    print("Posterior mean of sigma ", np.mean(sigmas))



if __name__ == '__main__':
    main()
