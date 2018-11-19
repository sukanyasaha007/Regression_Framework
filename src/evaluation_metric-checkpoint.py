import numpy as np

#mean square root
def msr_metric(y_test_in,y_pred_in):
    mean_sqr_error= np.mean(y_test_in-y_pred_in)**2
    return mean_sqr_error
#mean absolute error
def mar_metric(y_test_in,y_pred_in):
    mean_absolute_error= np.mean(abs(y_test_in-y_pred_in))
    return mean_absolute_error

#adjusted R square
def adjusted_R2score_calc(Xtest_in, r2score):
    n=Xtest_in.shape[0]
    k=Xtest_in.shape[1]
    adj_R2score= 1.0-(((1.0-float(r2score))*(float(n)-1.0))/(float(n)-float(k)-1.0))
    return adj_R2score