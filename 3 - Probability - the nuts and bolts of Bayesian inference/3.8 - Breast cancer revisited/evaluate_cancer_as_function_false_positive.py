from Breast_cancer_probability import *
from helper_functions import *


def compute_cancer_as_function_of_false_positive_rare(Cancer_probabilities, false_pos_linspace):
    """
        Computes p(cancer|+) using bayes' theorem where:
        p(cancer|+) = p(+|cancer)p(cancer) / (p(+|cancer)p(cancer) + p(+| not cancer)p(not cancer))
        for a range of values for p(+| not cancer) between [0,1] for a rare disease.
    """

    numerator = Cancer_probabilities.get_positive_given_cancer() * Cancer_probabilities.get_probability_cancer_rare()
    denominator = np.add(Cancer_probabilities.get_positive_given_cancer() * Cancer_probabilities.get_probability_cancer_rare(),
                         Cancer_probabilities.get_probability_not_cancer_rare() * false_pos_linspace)

    prob_cancer_as_function_of_false_positive = np.divide(
        numerator, denominator)

    return prob_cancer_as_function_of_false_positive


def compute_cancer_as_function_of_false_positive_common(Cancer_probabilities, false_pos_linspace):
    """
        Computes p(cancer|+) using bayes' theorem where:
        p(cancer|+) = p(+|cancer)p(cancer) / (p(+|cancer)p(cancer) + p(+| not cancer)p(not cancer))
        for a range of values for p(+| not cancer) between [0,1] for a common disease.
    """

    numerator = Cancer_probabilities.get_positive_given_cancer() * Cancer_probabilities.get_probability_cancer_common()
    denominator = np.add(Cancer_probabilities.get_positive_given_cancer() * Cancer_probabilities.get_probability_cancer_common(),
                         Cancer_probabilities.get_probability_not_cancer_rare() * false_pos_linspace)

    prob_cancer_as_function_of_false_positive = np.divide(
        numerator, denominator)

    return prob_cancer_as_function_of_false_positive


def evaluate_cancer_as_function_false_positive(Cancer_probabilities):

    prob_lower = 0
    prob_upper = 1

    false_positive_range = np.linspace(prob_lower, prob_upper, 100)

    prob_cancer_as_function_of_false_positive_rare = compute_cancer_as_function_of_false_positive_rare(
        Cancer_probabilities, false_positive_range)
    prob_cancer_as_function_of_false_positive_common = compute_cancer_as_function_of_false_positive_common(
        Cancer_probabilities, false_positive_range)

    plotter(prob_cancer_as_function_of_false_positive_rare, prob_cancer_as_function_of_false_positive_common, false_positive_range,
            "p(c|+) as a function of p(+|nc) for a rare (blue) and a common (green) disease", "p(+|nc) range", "probability of cancer given + test")
