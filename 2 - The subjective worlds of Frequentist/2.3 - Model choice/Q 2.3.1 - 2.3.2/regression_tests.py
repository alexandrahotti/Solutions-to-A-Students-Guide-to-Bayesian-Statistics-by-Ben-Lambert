import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def loadData( data_path ):

    data = pd.read_csv( data_path )

    X, Y = reshape_data( data )

    return X, Y


def reshape_data( data ):

    X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    return X,Y


def polynomial_regression( X, Y, degree):
    ''' Performs polynomial residual least square regression with a curve of an arbitary degree.
    '''

    polynomial_regression_values = np.polyfit(X.reshape(-1), Y.reshape(-1), degree)

    return polynomial_regression_values


def plot_results( X, Y, linear_model, quintic_model):
    """
    Creates a plit containing the scattered data, the linear model and the
    quintic model of the data.
    """

    plt.rcParams['figure.figsize'] = (7.0, 6.0)

    # A linspace used when the quintic model is plotted so that we get a smooth curve.
    no_data_points = len(X.reshape(-1))
    X_linspace = np.linspace( 1, no_data_points, 100 )

    plt.title( 'Least squares polynomials fit to 2D data points' )

    plt.scatter( X, Y, color = 'black' )
    plt.plot( X, np.polyval( linear_model, X ), 'blue' )
    plt.plot( X_linspace, np.polyval( quintic_model, X_linspace ), 'g--' )

    plt.show()


def evaluate_model(Y,X, model, degree):
    """
    RMSE is the square root of the average of the sum of the squares of residuals.
    R² score or the coefficient of determination explains how much the total variance
    of the dependent variable can be reduced by using the least square regression.
     """

    RMSE = calculate_RMSE(Y, X, model)
    r2_score_val = calculate_r2_score(Y, X, model)

    print('Evaluation of the '+ degree +' regression model:')
    print('The mean squared error of the residuals was: ', RMSE)
    print('The R² score or the coefficient of determination was: ', r2_score_val)
    print('\n')


def calculate_RMSE(Y, X, model):
    """
    RMSE is the square root of the average of the sum of the squares of residuals.
    """
    RMSE = mean_squared_error( Y, np.polyval( model, X ).reshape(-1) )

    return RMSE


def calculate_r2_score(Y, X, model):
    """
    R² score or the coefficient of determination explains how much the total variance
    of the dependent variable can be reduced by using the least square regression.
    """
    r2_score_val = r2_score(Y, np.polyval(model, X).reshape(-1) )

    return r2_score_val


def main():

    # Load the 2 dimensional data.
    data_path =  ( 'C:\\Users\\Alexa\\Desktop\\KTH\\EGET\\Bayesian_Ben_Lambert\\GITHUB\\Solutions-to-Problems-in-Bayesian-Statistics\\All_data\\subjective_overfitShort.csv' )
    X, Y = loadData( data_path )

    """ Fit the data to a linear model using least square error of the residuals. """
    degree = 1
    linear_model = polynomial_regression( X, Y, degree )

    """ Fit the data to a quintic model using least square error of the residuals. """
    degree = 5
    quintic_model = polynomial_regression( X, Y, degree )

    ''' Plot the data, the linear model and the quintic model'''
    plot_results( X, Y, linear_model, quintic_model )

    ''' Evaluate the performance of the models fit to the data.'''
    evaluate_model( Y, X, linear_model,'linear' )
    evaluate_model( Y, X, quintic_model,'quintic' )


if __name__ == '__main__':
    main()
