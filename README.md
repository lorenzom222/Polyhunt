# Polyhunt

## Commandlines:
```
polyhunt.py [-h] [-m M] [-k K] -g G [-f F] [-modelOutput MODELOUTPUT] [-af AF] [-i I] [-plots PLOTS]
```
## Example:
```
python polyhunt.py -m 40 -k 5 -g 1e-3 -f sampleData/Y -modelOutput HW1 -af T -i T -plots T
```

## 1. Method. 

The autofit method uses the concept of checking when the training and testing data begins to diverge due to overfitting. This is achieved by using the `diverging_index` function. The function calculates the point at which the testing data has the lowest `rms` before it starts to spike as a result of overfitting. The implementation of the `diverging_index` function is as follows:
```
def diverging_index(train_errors, test_errors):
    return np.where(test_errors == np.min(test_errors))[0][0] 
```
The test and training RMS are computed through cross-validation training on the minority and testing on the majority. The reason for this is the discovery that testing on the majority leads to a very biased fitting by the program or trying to fit too well, resulting in no visualization of overfitting in the test-train versus RMS model and a difficult gauge of overfitting to detect best polynomial degree. Training on the minority resolves this issue, eg. `k = 5`, Train: 20% and Test: 80%.

## 2. Regularization. 

Regularization was used in this method to prevent overfitting. The method tested regularization with small values of the regularization parameter `gamma`. The values tested were in the range of `gamma = 1e-2 to 1e-3`, and it was found that these values resulted in the lowest errors and the best fitting of the data.

Without regularization, the high degree polynomials can result in various floating point errors and computational mistakes, leading to significantly high errors. Regularization helps to constrain the complexity of the model and prevent overfitting, leading to better generalization to unseen data.


## 3. Model Stability. 

With regularization, the method yields consistent results. Even for the highest degree of polynomials `m = 200`, the variance in error is minimal. Only when viewing the `train-test RMS comparison` at a log scale can there be a clear visualization of overfitting, but ultimately the best-fitting polynomial is selected consistently. As stated before, only when the regularizer is not used is when the fitting and train-test graph start to get really crazy. But overall, model was stable and results will yield similar across others.


## 4. Results. 
All dataset results tested at `m = 40`, `k=5`. X had `gamma = 1e-3`, Y and Z had `gamma = 1e-2`. These gammas resulted in best fit for dataset.
The results of the autofit method for X, Y, and Z match the expected outcome on the labeled datasets to a certain extent. For X, the best degree obtained was 6 which is the expected degree of 6. The weights obtained are also close to the expected weights.

For Y, the best degree obtained was 9 which is the expected degree of 9. The weights obtained also show some differences from the expected weights, but they are still relatively close.

For Z, the best degree obtained was 4 which matches the expected degree of 4. The weights obtained are also close to the expected weights.

Overall, the results obtained seem to be correct, although there might be some small variations. These results show that the autofit method is capable of finding the best fitting polynomial to the data.


## 5. Notes. 

The polyhunt.py script has several optional arguments that can be used to control its behavior:

`-m M`: Specifies the degree of polynomial folds to use. Can either test at it without af or see if best polynomial lies when M is max. M is an integer value.

`-k K`: Specifies the number of times to repeat the cross-validation procedure. K is an integer value.

`-g G`: Specifies the value of gamma for the regularization term in the optimization problem. G is a float value.

`-f F`: Specifies the path to the input data file. F is a string value.

`-modelOutput MODELOUTPUT`: Specifies the path to the file where the final model will be saved. MODELOUTPUT is a string value.

`-af AF`: Specifies the autofit method to fit as best (or max at m) degree polynomial. AF is a boolean value.

`-i I`: See my personal information in console. I is an boolean value.

`-plots PLOTS`: Specifies whether or not to show plots. PLOTS is a boolean value.
