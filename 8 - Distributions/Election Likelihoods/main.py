from helper_functions import plotter
from Election_model import *


def main():

    Election_Likelihood_Model = Election_model()

    likelihoods, pB_range, pA_range = Election_Likelihood_Model.compute_multinomial_likelihood()

    plotter(pA_range, pB_range, likelihoods, "Multinomial Likelihood over pA and pB", "pA", "pB")


if __name__ == '__main__':
    main()
