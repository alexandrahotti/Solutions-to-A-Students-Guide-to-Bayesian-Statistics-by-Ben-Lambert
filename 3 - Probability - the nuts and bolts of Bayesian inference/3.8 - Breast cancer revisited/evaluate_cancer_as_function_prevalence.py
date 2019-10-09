from Breast_cancer_probability import *
from helper_functions import plotter_prevalence

import numpy as np


def compute_cancer_as_function_of_prevalence(Cancer_probabilities, prevalence_range):

    numerator = Cancer_probabilities.get_positive_given_cancer() * prevalence_range

    denominator = np.add(Cancer_probabilities.get_positive_given_cancer() * prevalence_range,
                         Cancer_probabilities.get_positive_given_not_cancer() * (1 - prevalence_range))

    prob_cancer_as_function_of_true_positive = np.divide(
        numerator, denominator)

    return prob_cancer_as_function_of_true_positive


def evaluate_cancer_as_function_prevalence(Cancer_probabilities):

    prob_lower = 0
    prob_upper = 1

    prevalence_range = np.linspace(prob_lower, prob_upper, 100)

    prob_cancer_as_function_of_prevalence = compute_cancer_as_function_of_prevalence(
        Cancer_probabilities, prevalence_range)

    plotter_prevalence(prob_cancer_as_function_of_prevalence, prevalence_range,
            "p(c|+) as a function of p(c)", "p(c) range", "probability of cancer given + test")
