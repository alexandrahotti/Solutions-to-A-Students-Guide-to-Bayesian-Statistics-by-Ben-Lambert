import numpy as np

def MLE_estimator(violent_crimes, population):

    avg_violent_crime = np.mean(violent_crimes)
    avg_population = np.mean(population)

    MLE_estimate = avg_violent_crime/avg_population

    return MLE_estimate
