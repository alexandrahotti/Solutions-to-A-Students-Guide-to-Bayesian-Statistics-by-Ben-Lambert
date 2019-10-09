from Breast_cancer_probability import *
from helper_functions import *


def compute_cancer_as_function_of_true_positive_rare(Cancer_probabilities, true_pos_linspace):
    """
        Computes p(cancer|+) using bayes' theorem where:
        p(cancer|+) = p(+|cancer)p(cancer) / (p(+|cancer)p(cancer) + p(+| not cancer)p(not cancer))
        for a range of values for p(+| cancer) between [0,1] for a rare disease.
    """

    numerator = true_pos_linspace * Cancer_probabilities.get_probability_cancer_rare()
    denominator = np.add(true_pos_linspace * Cancer_probabilities.get_probability_cancer_rare(),
                         Cancer_probabilities.get_probability_not_cancer_rare() * Cancer_probabilities.get_positive_given_not_cancer())
    prob_cancer_as_function_of_true_positive = np.divide(
        numerator, denominator)

    return prob_cancer_as_function_of_true_positive


def compute_cancer_as_function_of_true_positive_common(Cancer_probabilities, true_pos_linspace):
    """
        Computes p(cancer|+) using bayes' theorem where:
        p(cancer|+) = p(+|cancer)p(cancer) / (p(+|cancer)p(cancer) + p(+| not cancer)p(not cancer))
        for a range of values for p(+| cancer) between [0,1] for a common disease.
    """

    numerator = true_pos_linspace * Cancer_probabilities.get_probability_cancer_common()
    denominator = np.add(true_pos_linspace * Cancer_probabilities.get_probability_cancer_common(),
                         Cancer_probabilities.get_probability_not_cancer_common() * Cancer_probabilities.get_positive_given_not_cancer())
    prob_cancer_as_function_of_true_positive = np.divide(
        numerator, denominator)

    return prob_cancer_as_function_of_true_positive


def evaluate_cancer_as_function_true_positive(Cancer_probabilities):

    prob_lower = 0
    prob_upper = 1

    true_positive_range = np.linspace(prob_lower, prob_upper, 100)

    prob_cancer_as_function_of_true_positive_rare = compute_cancer_as_function_of_true_positive_rare(
        Cancer_probabilities, true_positive_range)
    prob_cancer_as_function_of_true_positive_common = compute_cancer_as_function_of_true_positive_common(
        Cancer_probabilities, true_positive_range)

    plotter(prob_cancer_as_function_of_true_positive_rare, prob_cancer_as_function_of_true_positive_common, true_positive_range,
            "p(c|+) as a function of p(+|c) for a rare (blue) and a common (green) disease", "p(+|c) range", "probability of cancer given + test")
