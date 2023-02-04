import numpy as np


def cross_validation(array, n_folds, step):
    array = np.array(array)
    for i in range(step, 0, -1):
        test_indices = np.arange(i-1, len(array), step)
        train_indices = np.delete(np.arange(len(array)), test_indices)
        test_fold = array[test_indices]
        train_fold = array[train_indices]
        # Do something with the test and train folds
        print(f"Test fold: {test_fold}")
        print(f"Train fold: {train_fold}")


array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
cross_validation(array, n_folds=5, step=5)
