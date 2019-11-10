from helper_functions import *
from Disease_outbreaks_model import *
import numpy as np
import statsmodels.api as sm
# import matplotlib.pyplot as plt



def main():

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\evaluation_discoveries.csv')
    evaluation_discoveries_data = loadData(data_path)

    years = get_column(evaluation_discoveries_data["time"])
    no_discoveries = get_column(evaluation_discoveries_data["discoveries"])


    # plotter(years, no_discoveries, '', 'time', 'no dicoveries')
    # plotter_histogram(no_discoveries, 'no discoveries / year', 'frequency', '')

    print('Variance: ' , np.var(no_discoveries))
    print('Mean: ' , np.mean(no_discoveries))

    autocorr = pd.Series(sm.tsa.acf(no_discoveries, nlags=10))
    plotter(range(0, 10), autocorr[1:len(autocorr)], '', 'lag', 'autocorrelation')


if __name__ == '__main__':
    main()
