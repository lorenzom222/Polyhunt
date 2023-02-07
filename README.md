# Polyhunt

Example:
```
polyhunt.py [-h] [-m M] [-k K] -g G [-f F] [-modelOutput MODELOUTPUT] [-af AF] [-i I] [-plots PLOTS]
```

```
python polyhunt.py -m 40 -k 10 -g 1e-5 -f sampleData/Y -modelOutput HW1 -af T -i T -plots T
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
