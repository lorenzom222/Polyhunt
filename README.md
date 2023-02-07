# Polyhunt

Example:
```
python polyhunt.py --m POLYNOMIAL --k NUM_FOLDS --g GAMMA --f FILENAME --i INFO --plots SHOW_PLOTS
```

For script with k-folds
```
python polyhunt.py --m 40 --k 5 --g 0.0 --f X --i True --plots P
```

For script without cross-validation
```
python polyhunt\(1\).py --m 40 --g 0.0 --f X --i True --plots P 
```

This one works good with $\lambda = 0.000001$ , MSE = 0.09

```
python polyhunt\(1\).py --m 25  --g 0.000001 --f Y --i True --plots P 

```

```
python polyhunt\(2\).py --m 20 --k 5  --g 0.001 --f X --plots P

```
## 1. Method. 

Briefly and specifically describe your autofit method. If you introduced
any custom parameters that are necessary for your autofit, describe them here.

## 2. Regularization. 

Did you use it? If you didn’t, probably your code will crash. What
values did you try? What value seemed to work the best?


## 3. Model Stability. 

Run your program a few different times on the data (or on different
subsets of the data) - how does the output change? Ideally you want a method that
always picks the same correct order every time. In practice, there will be some
variance. How stable are your predictions? If we run it a few times, will our results
agree with yours?


## 4. Results. 

Describe the results of your autofit method. Does it match the expected
outcome on the labeled datasets? What are the results for the second (unlabeled)
group of datasets? Do you believe these results are correct?


## 5. Notes. 

This section is optional. Include any notes to TAs regarding quirks of your
program, additional features, or other comments you believe may be helpful to know
when testing and grading.
