from helper_functions import plotter, plotter_discrete, plotter_histogram, loadData, get_column
from Epilepsy_model import *
import matplotlib.pyplot as plt



def main():

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\conjugate_epil.csv')
    epilepsy_data = loadData(data_path)

    epilepsy_counts = get_column(epilepsy_data["x"])

    print(len(epilepsy_counts))
    """ Q 9.2.3 The Gamma Posterior """
    a = 4
    b = 0.25

    eilepsy_model = Epilepsy_model(a, b)

    posterior, theta_range = eilepsy_model.posterior_gamma(epilepsy_counts)

    plotter(theta_range, posterior, "Gamma posterior over theta", "theta", "pdf")


if __name__ == '__main__':
    main()
