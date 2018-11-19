import os, sys
from directoryPath import parent_dir, data_dir, input_dir
import numpy as np
import pandas as pd
import sklearn as sk
from reading_data import read_data
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
	
from config_param import null_fraction, lower_bound, upper_bound, raw_data_train_valh5, raw_data_testh5, raw_data_test_csv, raw_data_train_val_csv, split_frac
from StatisticalModels import Ridge_baseline_model,Laso_baseline_model, Linear_baseline_model, enet_baseline_model,  linear_grid_search, ridge_grid_search, lasso_grid_search, BayesRidge_baseline_model, BayesRidge_grid_search
from MLModels import MLmodel_decisionTree_regressor, MLmodel_adaboostTree_regressor, MLmodel_randomForest_regressor, MLmodel_svr_regressor, decisionTree_regressor_gridsearchcv, adaBoost_regressor_gridsearchcv, randonForest_regressor_gridsearchcv, MLmodel_catboost_regressor, catb_regressor_gridsearchcv
from DLModels import DLmodel_regressor, DLmodel_regressor_gridsearch
from modelData_split import modeldata_split_train_test_validation, modeldata_split_train_test
from evaluation_metric import msr_metric, mar_metric
from data_summary import summary_function
from data_prep import creating_label_features
from lime_ModelExplainer import lime_explainer




#Variable selection
""" 
#Test data feature selection
raw_data_test_csv= raw_data_test_csv #mentioned in config_param
df_test=read_data(input_dir + raw_data_test_csv)

df_test=remove_null(df_test,null_fraction) 
df_test=feature_selection(df_test, lower_bound, upper_bound)
df_test.to_hdf(data_dir + 'test.h5', 'test')

#Train/val data feature selection
raw_data_train_val_csv= raw_data_train_val_csv
df_test=read_data(input_dir + raw_data_train_val_csv)
df_train_val=remove_null(df_hd,null_fraction) 
df_train_val=feature_selection(df_train_val, lower_bound, upper_bound)
df_train_val.to_hdf(data_dir + 'train_val.h5', 'train')
""" 
#data summary
#summary_function(df_test)

#reading data h5 format data
print("Reading data....")
raw_data_train_val = raw_data_train_valh5
raw_data_test = raw_data_testh5


#creating X, y train validation data set 
df_train_val=read_data( data_dir + raw_data_train_val)
df_train_val_X, df_train_val_y=creating_label_features(df_train_val)

#creating X, y test data set 
df_test=read_data( data_dir + raw_data_test)
X_test,y_test= creating_label_features(df_test)





#train validation set split
print ("Creating train, validation and test data set ....")
X_train,  X_val,  y_train, y_val=modeldata_split_train_test(df_train_val_X, df_train_val_y, split_frac)
#X_train, X_test, X_val,  y_train, y_test, y_val=modeldata_split_train_test_validation(df_train_val_X, df_train_val_y, 0.2, 0.2)

print ("Modeling....")
#~~~~~~~~~~~~~~~~~~~base regression models~~~~~~~~~~~

print (Ridge_baseline_model(X_train, y_train, X_test, y_test, True, X_train.iloc[150,:]))
#print (Laso_baseline_model(X_train, y_train, X_test, y_test))
#print(enet_baseline_model(X_train, y_train, X_test, y_test))
#print (BayesRidge_baseline_model(X_train, y_train, X_test, y_test))
#~~~~~~~~~~~~~~~~~~~~~gridsearchcv~~~~~~~~~~~~~~
#print (lasso_grid_search(X_train, y_train, X_test, y_test, True, X_train.iloc[150,:]))
#print (ridge_grid_search(X_train, y_train, X_test, y_test))
#print (linear_grid_search(X_train, y_train, X_test, y_test, True, X_train.iloc[150,:]))
#print (BayesRidge_grid_search(X_train, y_train, X_test, y_test))


#~~~~ ML models Tree resgressors~~~~~~~~~~~~~~

#print (MLmodel_randomForest_regressor(X_train, y_train, X_test, y_test))
#print (MLmodel_decisionTree_regressor(X_train, y_train, X_test, y_test, True, X_train.iloc[150,:]))
#print (MLmodel_adaboostTree_regressor(X_train, y_train, X_test, y_test))
#print (MLmodel_svr_regressor(X_train, y_train, X_test, y_test))
#print (MLmodel_lgbm_regressor(X_train, y_train, X_test, y_test))
#print (MLmodel_catboost_regressor(X_train, y_train, X_test, y_test))

#print(randonForest_regressor_gridsearchcv(X_train, y_train, X_test, y_test))
#print(adaBoost_regressor_gridsearchcv(X_train, y_train, X_test, y_test))
#print(decisionTree_regressor_gridsearchcv(X_train, y_train, X_test, y_test, True, X_train.iloc[150,:]))
#print (lgbm_regressor_gridsearchcv(X_train, y_train, X_test, y_test))
#print (catb_regressor_gridsearchcv(X_train, y_train, X_test, y_test))

#~~~~~~~~~~~~~~~~~Deep learning regressors~~~~~~~~
#print(DLmodel_regressor(X_train, y_train, X_test, y_test))
#print (MLmodel_xgb_regressor(X_train, y_train, X_test, y_test))  
#print (DLmodel_regressor_gridsearch(X_train.iloc[:500000,:], y_train.iloc[:500000], X_test.iloc[:10000,:], y_test.iloc[:10000]))