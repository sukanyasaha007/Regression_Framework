import numpy
import pandas
import numpy as np
import sys
import time
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
	
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from evaluation_metric import adjusted_R2score_calc
from log_outputs import log_record_result
from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer


#~~~~~~~~~~~~~~~~base model~~~~~~~~~~~~~~~~~~~~
def DLmodel_baseline():
# create model
	model = Sequential()
	model.add(Dense(61, input_dim=61, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def DLmodel_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	estimator = KerasRegressor(build_fn=DLmodel_baseline, epochs=20, batch_size=5, verbose=10)
	seed = 23 
	numpy.random.seed(seed) 
	estimator.fit(Xtrain_in, ytrain_in) 
	y_test_pred             =estimator.predict(Xtest_in)
	y_train_pred            =estimator.predict(Xtrain_in)
	score_test              = r2_score(y_test_pred, ytest_in)
	score_train             =r2_score(y_train_pred, ytrain_in)
	adj_Rscore_train        = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test         = adjusted_R2score_calc(Xtest_in, score_test)
	time_end                =time.time() - start_time
	mrs_train              = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test               = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, estimator, "Keras_base")	
	time_end                =time.time() - start_time
	log_record_result("Keras base model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, estimator, "Keras_base") #plots residual
	return "Keras base model", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test) 

 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~Gridsearch~~~~~~~~~~~~~~~~~~~~~~~~~    

def DLmodel_model():
    model = Sequential()
    model.add(Dense(61, input_dim=61, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))   
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def DLmodel_regressor_gridsearch(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time          = time.time()
	estimator           = KerasRegressor(build_fn=DLmodel_model,  verbose=10)
	batch_size          = [20, 50, 100]
	epochs              = [50, 100, 200]
	param_grid          = dict(batch_size=batch_size, epochs=epochs)#,  neurons=neurons)
	grid                = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
	grid_result         = grid.fit(Xtrain_in, ytrain_in)
	seed = 23 
	np.random.seed(seed) 
	estimator.fit(Xtrain_in, ytrain_in) 

	y_test_pred          =grid_result.predict(Xtest_in)
	y_train_pred         =grid_result.predict(Xtrain_in)
	score_test           = r2_score(y_test_pred, ytest_in)
	score_train          =r2_score(y_train_pred, ytrain_in)
	adj_Rscore_train     = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test      = adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters      =grid.best_params_
	time_end             =time.time() - start_time
	mrs_train           = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test            = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "Keras_grid")
	time_end=time.time() - start_time
	log_record_result("Keras base model gridsearch", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "Keras_grid") #plots residual
	return "Keras base model gridsearch", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)


