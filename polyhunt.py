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


# def sweep(X, m, lambda_):
#     """
#     Sweep up to max polynomial (m).

#     Parameters:
#     X : Training data input. (trainPath)
#     m : Given polynomial.
#     lambda_ : Regularization parameter.

#     Returns:
#     weight_list : List of best regularized weights per polynomial.
#     """
#     weight_list = []
#     for i in range(m+1):
#         phi_m = create_phi(X, i)
#         weight = regularized_linear_regression(phi_m, t, 0)
#         weight_list.append(weight)
#     return weight_list


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


def kfold_cv_k(x, t, M, k):
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
    # folds = np.array_split(data, k)
    error_list = []
    error_TRAINlist = []

    # print(f"data shape = {data.shape}")
    for i in range(k):
        validation_indices = np.array([j for j in range(len(x)) if j % k == i])
        # print(f"k={k}, validation: {validation_indices.shape}")
        # print(validation_indices)
        validation_fold = data[validation_indices]
        # print(f"valid f = {validation_fold.shape}")
        training_indices = np.array([j for j in range(len(x)) if j % k != i])
        # print(f"k={k}, training: {training_indices.shape}")

        training_folds = data[training_indices]
        x_train, t_train = training_folds[:, 0], training_folds[:, 1]
        phi_train = create_phi(x_train, M)
        w_train = regularized_linear_regression(phi_train, t_train, 0)
        ######
        Y_train = prediction(x_train, phi_train, w_train)
        error_TRAIN = errorfunction(Y_train, t_train)
        error_TRAINlist.append(error_TRAIN)

        ######
        # print(f"w_train = {w_train}")

        x_test, t_test = validation_fold[:, 0], validation_fold[:, 1]
        phi_test = create_phi(x_test, M)
        y_test = prediction(x_test, phi_test, w_train)

        error = errorfunction(y_test, t_test)
        error_list.append(error)
    avg_error = np.mean(error_list)
    avg_TRAINerror = np.mean(error_TRAINlist)

    return avg_error, error_list, avg_TRAINerror, error_TRAINlist


def solve_curve_fitting_k(x, t, M, k, gamma):
    phi = np.array(create_phi(x, M))
    w = regularized_linear_regression(phi, t, gamma)
    y = prediction(x, phi, w)
    error = errorfunction(y, t)
    rms = np.sqrt(2*error/len(x))
    k_fold = kfold_cv_k(x, t, M, k)
    avg_error = k_fold[0]
    errorlist = k_fold[1]
    avg_TRAINerror = k_fold[2]
    error_TRAINlist = k_fold[3]
    # avg_error, errorlist, , error_TRAINlist = kfold_cv_k(x, t, M, k)
    k_rms = np.sqrt(2*avg_error/len(x))
    k_rms_TRAIN = np.sqrt(2*avg_TRAINerror/len(x))

    return w, y, error, rms, avg_error, k_rms, errorlist, k_rms_TRAIN, error_TRAINlist
    # return rms


def plot_tt_k(x, t, m, ratio, k, gamma, plots):
    # x_train, t_train, x_test, t_test = split_data(x, t, ratio)
    M = np.arange(0, m)
    train_rms = []
    test_rms = []
    for i in M:
        rms = solve_curve_fitting_k(x, t, i, k, gamma)
        train_rms.append(rms[7])
        test_rms.append(rms[5])
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

    parser.add_argument('--k', type=int,
                        help='integer - folds for cross validation')

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
k = args.k
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
    plt.title('MSE: {:.2f}'.format(avg_error) +
              ' and K_RMS: {:.2f}'.format(k_rms))
    plt.show()


# print('Coefficients:', w)
# print('Predicted values:', y)
print('Mean Squared Error:', avg_error)
print('Root-Mean-Squared:', k_rms)

train_rms, test_rms = plot_tt_k(X, t, M, ratio, k, gamma, plots)


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
