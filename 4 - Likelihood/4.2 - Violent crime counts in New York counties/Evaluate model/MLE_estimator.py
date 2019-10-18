import numpy as np
from scipy.stats import poisson

def MLE_estimator(violent_crimes, population):

    avg_violent_crime = np.mean(violent_crimes)
    avg_population = np.mean(population)

    MLE_estimate = avg_violent_crime/avg_population

    return MLE_estimate


def generate_data_samples(MLE_parameter, no_samples):

    generated_samples = poisson.rvs(mu=MLE_parameter*no_samples, size=no_samples)

    return generated_samples

def calculate_per_capita_crime_rate(violent_crime_count, population):
    per_capita_crime_rate = np.divide(violent_crime_count,population)

    return per_capita_crime_rate
