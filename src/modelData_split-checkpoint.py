import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

#this function takes in Y and X data sets and split them to three sets: train/validation and test
def modeldata_split_train_test_validation(df_in_X, df_in_y, test_size, validation_size):
    '''
    takes in Y and X data sets and split them to three sets
    Parameters: 
    df_in_X, df_in_y, test_size, validation_size
    return X_train, X_test, X_val,  y_train, y_test, y_val
    '''
        
    val_adjusted_size=float(validation_size)/(1.0-float(validation_size))
    X_train, X_test, y_train, y_test = train_test_split(df_in_X, df_in_y, test_size=test_size, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_adjusted_size, random_state=1)
    return X_train, X_test, X_val,  y_train, y_test, y_val

#This takes X, y data sets and split the data to train and test
def modeldata_split_train_test(df_in_X, df_in_y, test_size):
    '''
    takes in Y and X data sets and split them to two sets
    Parameters: 
    df_in_X, df_in_y, test_size
    return X_train, X_test, X_val,  y_train, y_test, y_val
    '''
    X_train, X_test, y_train, y_test = train_test_split(df_in_X, df_in_y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test
  