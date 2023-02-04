import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import argparse
import sys


def load_data(filepath):
    """    
    Parameters:
    filepath (cvs): training data.

    Returns:
    x (np.array) : inputs, attributes.
    y (np.array) : outputs.

    """
    X, y = [], []
    with open(filepath, 'r') as file:
        for line in file:
            x, y_ = line.strip().split(',')
            X.append(float(x))
            y.append(float(y_))
    return np.array(X), np.array(y)


def load_sol_data(filepath):
    """    
    Parameters:
    filepath (cvs): solution data.

    Returns:
    m (int) : polynomial degree.
    w (np.array) : weights.

    """
    with open(filepath, 'r') as file:
        m = int(file.readline().strip())
        w = []
        for line in file:
            w_ = float(line.strip())
            w.append(w_)
    return m, np.array(w)


def create_phi(x, degree):
    """
    This function creates a matrix of x values raised to powers.
    :param x: numpy array of input data
    :param degree: degree of the polynomial
    :return: matrix of x values raised to powers
    """
    matrix = np.zeros((len(x), degree + 1))
    for i in range(len(x)):
        for j in range(degree + 1):
            matrix[i][j] = x[i] ** j
    return matrix


def regularized_linear_regression(phi, t, lambda_):
    """
    Find the best fitting regularized weights for a linear model using equation 3.28 from PRML.

    Parameters:
    X : Training data input. (trainPath)
    y : Training data output. (modelOutput)
    lambda_ : Regularization parameter.

    Returns:
    w : Best regularized weights.
    """
    i, size = phi.shape
    phi_t = phi.transpose()
    phit_phi = np.matmul(phi_t, phi)
    id = np.identity(size)
    phit_phi += lambda_ * id
    phit_phi_inv = np.linalg.inv(phit_phi)
    w_reg = np.matmul(phit_phi_inv, phi_t)
    w = np.matmul(w_reg, t)
    return w


def sweep(X, m, lambda_):
    """
    Sweep up to max polynomial (m).

    Parameters:
    X : Training data input. (trainPath)
    m : Given polynomial.
    lambda_ : Regularization parameter.

    Returns:
    weight_list : List of best regularized weights per polynomial.
    """
    weight_list = []
    for i in range(m+1):
        phi_m = create_phi(X, i)
        weight = regularized_linear_regression(phi_m, t, 0)
        weight_list.append(weight)
    return weight_list


def prediction(X, phi, w):
    """
    Compute prediction for polynomial regression.

    Parameters:
    X (np.array) : Training data input.
    phi (matrix) : Matrix of x values raised to powers.
    w (np.array) : Best regularized weights.

    Returns:
    y(X, phi, w) : prediction.
    """

    return np.dot(phi, w)


def small_phi(phi, r):
    return phi[r]


def errorfunction(y, t):
    error = 0.0
    for i in range(len(y)):
        error += (y[i]-t[i])**2

    return 1/2*error


def split_data(x, t, ratio):
    # Shuffle data
    shuffle_index = np.random.permutation(len(x))
    x = x[shuffle_index]
    t = t[shuffle_index]

    # Split data into training and test sets
    split_index = int(len(x) * ratio)
    x_train = x[:split_index]
    t_train = t[:split_index]
    x_test = x[split_index:]
    t_test = t[split_index:]

    return x_train, t_train, x_test, t_test


def rms(error, N):
    return np.sqrt((2*error)/N)


def prediction(X, phi, w):
    """
    Compute prediction for polynomial regression.

    Parameters:
    X (np.array) : Training data input.
    phi (matrix) : Matrix of x values raised to powers.
    w (np.array) : Best regularized weights.

    Returns:
    y(X, phi, w) : prediction.
    """

    return np.dot(phi, w)


def errorfunction(y, t):
    error = 0.0
    for i in range(len(y)):
        error += (y[i]-t[i])**2

    return 1/2*error


def solve_curve_fitting(x, t, M, gamma):
    phi = np.array(create_phi(x, M))
    w = regularized_linear_regression(phi, t, gamma)
    y = prediction(x, phi, w)
    error = errorfunction(y, t)
    rms = np.sqrt(2*error/len(x))

    return w, y, error, rms


def split_data(x, t, ratio):
    # Shuffle data
    shuffle_index = np.random.permutation(len(x))
    x = x[shuffle_index]
    t = t[shuffle_index]

    # Split data into training and test sets
    split_index = int(len(x) * ratio)
    x_train = x[:split_index]
    t_train = t[:split_index]
    x_test = x[split_index:]
    t_test = t[split_index:]

    return x_train, t_train, x_test, t_test


def plot_tt_k(x, t, m, ratio, gamma, plots):
    x_train, t_train, x_test, t_test = split_data(x, t, ratio)
    M = np.arange(0, m)
    train_rms = []
    test_rms = []
    for i in M:
        rms_train = solve_curve_fitting(x_train, t_train, i, gamma)
        train_rms.append(rms_train[3])
        rms_test = solve_curve_fitting(x_test, t_test, i, gamma)
        test_rms.append(rms_test[3])
    if(plots):
        plt.plot(M, train_rms, '-o', label='Train RMS')
        plt.plot(M, test_rms, '-o', label='Test RMS')
        plt.xlabel('M (degree)')
        plt.ylabel('RMS')
        plt.legend()
        plt.show()
    return train_rms, test_rms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--m', type=int,
                        help='integer - polynomial order (or maximum in autofit mode)')

    # parser.add_argument('--k', type=int,
    #                     help='integer - folds for cross validation')

    parser.add_argument('--g', type=float, required=True,
                        help='float - regularization constant (use a default of 0')

    parser.add_argument('--f', type=str,
                        help='the path of the file')

    # parser.add_argument('modelOutput', type=str, required=True,
    #                     help='integer - polynomial order (or maximum in autofit mode)')

    # parser.add_argument('autofit', type=bool, required=True,
    #                     help='integer - polynomial order (or maximum in autofit mode)')
    parser.add_argument('--i', type=bool,
                        help='Name and contact information')
    parser.add_argument('--plots', type=bool,
                        help='Show plots')

args = parser.parse_args()
M = args.m
# k = args.k
gamma = args.g

folder = "sampleData/"
input = load_data(folder+args.f)
X, t = input
i = args.i
plots = args.plots

if(i):
    print("|--------------------------------|")
    print("         Lorenzo Mendoza")
    print("      lmendoz4@u.rochester.edu")
    print("            POLYHUNT")
    print("|--------------------------------|\n")


ratio = 8/10
fit = solve_curve_fitting(X, t, M, gamma)
w = fit[0]
y = fit[1]
error = fit[2]
rms = fit[3]


if(plots):
    plt.plot(X, t, 'o', label='Target')
    plt.plot(X, y, '-', label='Prediction')
    plt.legend()
    plt.title('MSE: {:.2f}'.format(error) +
              ' and K_RMS: {:.2f}'.format(rms))
    plt.show()


# print('Coefficients:', w)
# print('Predicted values:', y)
print('Mean Squared Error:', error)
print('Root-Mean-Squared:', rms)

train_rms, test_rms = plot_tt_k(X, t, M, ratio, gamma, plots)


def when_diverges(train, test):
    for i in range(5, len(train)):
        if train[i] > train[i-1] and train[i] > test[i]:
            return i
        if test[i] > test[i-1] and test[i] > train[i]:
            return i

    return None


index = when_diverges(train_rms, test_rms)
if index == None:
    print('At Max Best Fit with input for M:', M)

else:
    print('Best Polynomia Degree (M):', index - 1)


"""
i
"""

"""
Command Line:
python polyhunt\(1\).py --m 25 --g 0.1 --f Z --i True --plots P

"""
