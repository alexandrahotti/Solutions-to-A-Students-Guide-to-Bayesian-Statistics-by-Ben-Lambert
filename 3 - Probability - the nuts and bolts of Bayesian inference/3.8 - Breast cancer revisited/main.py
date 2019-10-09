from Breast_cancer_probability import *
from evaluate_cancer_as_function_true_positive import evaluate_cancer_as_function_true_positive
from evaluate_cancer_as_function_false_positive import evaluate_cancer_as_function_false_positive
from evaluate_cancer_as_function_prevalence import evaluate_cancer_as_function_prevalence


def main():

    Cancer_probabilities = Breast_cancer_probability()

    evaluate_cancer_as_function_true_positive(Cancer_probabilities)
    evaluate_cancer_as_function_false_positive(Cancer_probabilities)
    evaluate_cancer_as_function_prevalence(Cancer_probabilities)


if __name__ == '__main__':
    main()
