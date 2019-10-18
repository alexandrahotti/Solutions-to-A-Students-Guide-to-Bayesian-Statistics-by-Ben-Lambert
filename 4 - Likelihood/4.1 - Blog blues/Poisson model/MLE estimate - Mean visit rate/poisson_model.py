import numpy as np


def compute_MLE_mean_time_between_visits(time_between_visits):
    """ Computes the Maximum likelihood estimate of the first time visit rate
        to a blog.

        The etsimate is given by: \hat{\lambda} = \frac{1}{\bar{T}}
        where lambda is the first time visit rate and \bar{T} is the
        average time between visits.
    """
    avg_time_between_visits = np.mean(time_between_visits)
    first_time_visit_rate = 1 / avg_time_between_visits

    return first_time_visit_rate


def log_likelihood_as_function_of_mean_visit_rate(time_between_visits):
    """
        Computes the log likelihood as a function of the mean visit rate.
    """
    # A lower and upper limit for the mean visit rate per minute.
    mean_visit_rate_lower = 0.0001
    mean_visit_rate_upper = 10

    mean_visit_rate_range = np.linspace(
        mean_visit_rate_lower, mean_visit_rate_upper, 100)

    log_likelihoods = compute_log_likelihoods(
        time_between_visits, mean_visit_rate_range)

    return log_likelihoods, mean_visit_rate_range


def compute_log_likelihoods(time_between_visits, lambda_range):

    # Actually number of "inbetween visits".
    no_visits = len(time_between_visits)

    mean_time_between_visits = np.mean(time_between_visits)

    log_likelihoods = no_visits * \
        np.log(lambda_range) - (no_visits *
                                lambda_range * mean_time_between_visits)

    return log_likelihoods
