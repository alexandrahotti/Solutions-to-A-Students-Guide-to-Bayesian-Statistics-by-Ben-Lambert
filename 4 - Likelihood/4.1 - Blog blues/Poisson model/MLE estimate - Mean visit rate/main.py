import pandas as pd
from time_between_blog_visits import *
from helper_functions import plotter


def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    data_values = reshape_data(data_frame)

    return data_values


def reshape_data(data_frame):

    data_values = data_frame.values.reshape(-1)

    return data_values


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\likelihood_blogVisits.csv')
    time_between_visits = loadData(data_path)

    """ Solution to question 4.1.4 """
    MLE_first_time_visit_rate = compute_MLE_mean_time_between_visits(
        time_between_visits)

    print("The MLE of the first time visit rate for a blog, using 50 data points.")
    print("MLE estimate for lambda: ", MLE_first_time_visit_rate)

    """ Solution to question 4.1.5 """
    log_likelihoods, mean_visit_rates = log_likelihood_as_function_of_mean_visit_rate(
        time_between_visits)

    log_likelihood_MLE = compute_log_likelihoods(
        time_between_visits, MLE_first_time_visit_rate)

    plotter(log_likelihoods, mean_visit_rates, MLE_first_time_visit_rate, log_likelihood_MLE,
            "The log likelihood as a function of the rate between first time visits", "Avg no. visits per min", "log likelihood")


if __name__ == '__main__':
    main()
