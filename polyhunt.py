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


def save_parameters(filepath, m, w):
    if filepath:
        # Save the best fit parameters to the filepath
        with open(filepath, 'w') as f:

            f.write(f'Polynomial Degree: {m}\nWeights: {w}')
    else:
        # Do not save the parameters if filepath is not supplied
        pass


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


def prediction(phi, w):
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
    d = y-t
    d = d**2
    d = np.sum(d)
    d = np.sqrt(d/len(y))

    return d


def kfold_cv_k(x, t, M, k, gamma):
    """
    K-Fold Cross Validation

    Parameters:
    x (np.array): inputs, attributes.
    t (np.array): outputs.
    M (int): polynomial degree.
    k (int): number of folds.

    Returns:
    avg_error (float): average root mean squared error.
    """
    data = np.concatenate((x.reshape(-1, 1), t.reshape(-1, 1)), axis=1)
    error_list = []
    error_TRAINlist = []

    for i in range(k):
        training_indices = np.array([j for j in range(len(x)) if j % k == i])
        training_folds = data[training_indices]

        test_indices = np.array([j for j in range(len(x)) if j % k != i])
        test_fold = data[test_indices]
        x_train, t_train = training_folds[:, 0], training_folds[:, 1]
        phi_train = create_phi(x_train, M)
        w_train = regularized_linear_regression(phi_train, t_train, gamma)
        Y_train = prediction(phi_train, w_train)
        error_TRAIN = errorfunction(Y_train, t_train)
        error_TRAINlist.append(error_TRAIN)

        x_test, t_test = test_fold[:, 0], test_fold[:, 1]
        phi_test = create_phi(x_test, M)
        y_test = prediction(phi_test, w_train)

        error = errorfunction(y_test, t_test)
        error_list.append(error)

    avg_error = np.mean(error_list)
    avg_TRAINerror = np.mean(error_TRAINlist)

    test_rms = avg_error
    train_rms = avg_TRAINerror

    return test_rms, error_list, train_rms, error_TRAINlist


def solve_curve_fitting_k(x, t, M, k, gamma):

    phi = np.array(create_phi(x, M))
    w = regularized_linear_regression(phi, t, gamma)
    y = prediction(phi, w)
    error = errorfunction(y, t)
    rms = np.sqrt(2*error/len(x))
    k_fold = kfold_cv_k(x, t, M, k, gamma)
    avg_error = k_fold[0]
    errorlist = k_fold[1]
    avg_TRAINerror = k_fold[2]
    error_TRAINlist = k_fold[3]
    k_rms = np.sqrt(2*avg_error/len(x))
    k_rms_TRAIN = np.sqrt(2*avg_TRAINerror/len(x))

    return w, y, error, rms, avg_error, k_rms, errorlist, k_rms_TRAIN, error_TRAINlist


def plot_tt_k(x, t, m, k, gamma, plots):
    M = np.arange(0, m)
    train_rms = []
    test_rms = []
    for i in M:
        cross_val = kfold_cv_k(x, t, i, k, gamma)
        train_rms.append(cross_val[2])
        test_rms.append(cross_val[0])

    if(plots):
        plt.plot(M, train_rms, '-o', label='Train RMS')
        plt.plot(M, test_rms, '-o', label='Test RMS')
        plt.yscale('log')
        plt.xlabel('M (degree)')
        plt.ylabel('RMS')
        plt.legend()
        plt.show()
    return train_rms, test_rms


def diverging_index(train_errors, test_errors):
    return np.where(test_errors == np.min(test_errors))[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', type=int,
                        help='integer - polynomial order (or maximum in autofit mode)')

    parser.add_argument('-k', type=int,
                        help='integer - folds for cross validation')

    parser.add_argument('-g', type=float, required=True,
                        help='float - regularization constant (use a default of 0')

    parser.add_argument('-f', type=str,
                        help='the path of the file')

    parser.add_argument('-modelOutput', type=str,
                        help='string - a filepath where the best fit parameters will be saved, if this is not supplied, then you do not have to output any model parameters')

    parser.add_argument('-af', type=bool,
                        help='integer - polynomial order (or maximum in autofit mode)')
    parser.add_argument('-i', type=bool,
                        help='Name and contact information')
    parser.add_argument('-plots', type=bool,
                        help='Show plots')

args = parser.parse_args()
M = args.m
k = args.k
gamma = args.g

# folder = "sampleData/"
input = load_data(args.f)
X, t = input
modelOutput = args.modelOutput

af = args.af

i = args.i
plots = args.plots

if(i):
    print("|--------------------------------|")
    print("         Lorenzo Mendoza")
    print("      lmendoz4@u.rochester.edu")
    print("            POLYHUNT")
    print("|--------------------------------|\n")


# ratio = 8/10
fit = solve_curve_fitting_k(X, t, M, k, gamma)
w = fit[0]
y = fit[1]
error = fit[2]
rms = fit[3]
avg_error = fit[4]
k_rms = fit[5]

if(plots):
    plt.plot(X, t, 'o', label='Target')
    plt.plot(X, y, '-', label='Prediction')
    plt.legend()
    plt.title('M: {:.0f}'.format(
        M)+' | Error: {:.2f}'.format(avg_error) + ' and K_RMS: {:.2f}'.format(k_rms))

    plt.show()


train_rms, test_rms = plot_tt_k(X, t, M, k, gamma, plots)


best_M = diverging_index(train_rms, test_rms)

if(af):
    best_fit = solve_curve_fitting_k(X, t, best_M, k, gamma)
    best_weight = best_fit[0]
    best_error = best_fit[4]
    best_k_rms = best_fit[5]
    print('Best:', best_M)
    print('Best-Weight:', best_weight)
    # print('Best-Mean Squared Error:', best_error)
    # print('Best-Root-Mean-Squared:', best_k_rms)

else:
    print('Best:', M)
    print('Weights:', w)
    # print('Predicted values:', y)
    # print('Mean Squared Error:', avg_error)
    # print('Root-Mean-Squared:', k_rms)

if modelOutput:
    if af:
        save_parameters(modelOutput, best_M, best_weight)
    else:
        save_parameters(modelOutput, M, w)
