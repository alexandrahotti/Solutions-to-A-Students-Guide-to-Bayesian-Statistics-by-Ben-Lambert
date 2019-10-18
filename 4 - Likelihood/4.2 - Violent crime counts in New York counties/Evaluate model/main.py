import pandas as pd
from helper_functions import plotter
from MLE_estimator import MLE_estimator, generate_data_samples, calculate_per_capita_crime_rate
import numpy as np

def loadData(data_path):

    data_frame = pd.read_csv(data_path)

    return data_frame


def reshape_data(data_column):

    data_values = data_column.values.reshape(-1)

    return data_values

def convert_str_to_int(string_list):

    string_list = [ x.replace(',','') for x in string_list ]
    float_list = [ int(x) for x in string_list ]

    return float_list


def main():
    """ Load and pre-process data. """

    data_path = ('C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\likelihood_NewYorkCrimeUnemployment.csv')
    crime_dataframe = loadData(data_path)

    population =  convert_str_to_int(reshape_data(crime_dataframe["Population"]))
    violent_crime_count = convert_str_to_int(reshape_data(crime_dataframe["Violent_crime_count"]))

    plotter(population, violent_crime_count, "population", "violent crime count")

    MLE_estimate = MLE_estimator(violent_crime_count, population)

    print("The MLE is:", MLE_estimate)

    """ Problem 4.2.4 """

    generated_samples = generate_data_samples(0.01, len(population))

    per_capita_crime_rate = calculate_per_capita_crime_rate(violent_crime_count, population)

    print(generated_samples)
    print(per_capita_crime_rate)

if __name__ == '__main__':
    main()
