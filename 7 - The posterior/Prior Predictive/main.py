import pandas as pd
from helper_functions import plotter
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

import statsmodels.api as sm

import statistics
from sklearn.metrics import mean_squared_error


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


def calculate_standard_error(x_val, y_val):

    X = sm.add_constant(x_val)

    ols = sm.OLS(y_val, X)
    ols_result = ols.fit()

    # Now you have at your disposition several error estimates, e.g.
    print(ols_result.summary())


def calculate_RMSE(gdps, mortalities, intercept, slope):
    """
        Calculates the RMSE (std of teh residuals) of our modeled mortaility values and the
        actual infant mortality values.
    """

    # Calculate the mortaility values for our model.
    mortalities_model = [gdp * slope + intercept for gdp in np.log(gdps)]

    # Compare the modeled values with the actual values by comouting the MSE of the residuals.
    MSE = mean_squared_error(np.log(mortalities), mortalities_model)

    # Calculate the RMSE (std of the residuals) by taking the square root of the MSE.
    print('RMSE ', np.sqrt(MSE))


def sample_alpha_prior():
    """
        Sample from the prior distribution of alpha.
        A normal distribution centered at 0 with a std of 10.
    """

    return np.random.normal(7, 3, 1)

def sample_beta_prior():
    """
        Sample from the prior distribution of beta.
        A normal distribution centered at 0 with a std of 10.
    """

    return np.random.normal(-0.5, 1, 1)

def sample_sigma_prior():
    """
        Sample from the prior distribution of sigma.
        A half normal distribution centered at 0 with a std of 5.
    """

    return stats.halfnorm.rvs(0, 1, 1)

def estimate_prior_predictive_distribution(gdps):

    mortalities_prior_predictive = []
    no_iterations = 50
    for it in range(no_iterations):

        """ Sample alpha and beta parameters from their priors."""
        alpha_i = sample_alpha_prior()[0]
        beta_i = sample_beta_prior()[0]
        sigma_i = sample_sigma_prior()[0]
        
        """ Now condition the likelihood on the sampled parameters and then
            generate a sample of the mortality rate from the likelihood. """
        mortalities_model_i = []
        i = 0
        for gdp in gdps:
            mean_i = alpha_i + beta_i * gdp
            mortality_i = np.random.normal(mean_i, sigma_i, 1)[0]
            # print('mor i ',mortality_i)
            mortalities_model_i.append(mortality_i)
            if i == 0:
                print(mortality_i)
                i = 1
        i = 0

        mortalities_prior_predictive.append(mortalities_model_i)


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\posterior_gdpInfantMortality.csv')
    GDP_data = loadData(data_path)

    mortality_data = get_column(GDP_data["infant.mortality"])
    gdp_data = get_column(GDP_data["gdp"])

    # Remove data points with missing data i.e. NaN values.
    gdps, mortalities = remove_missing_data(gdp_data, mortality_data)

    estimate_prior_predictive_distribution(np.log(gdps))


if __name__ == '__main__':
    main()
