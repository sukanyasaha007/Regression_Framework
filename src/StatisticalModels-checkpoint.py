'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import BayesianRidge
from evaluation_metric import adjusted_R2score_calc
from log_outputs import log_record_result
from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer

def Ridge_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	ridgeRegr          = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver="auto", random_state=None)      
	ridgeRegr.fit(Xtrain_in, ytrain_in)
	score_test         =ridgeRegr.score(Xtest_in, ytest_in)
	score_train        =ridgeRegr.score(Xtrain_in, ytrain_in)
	adj_Rscore_train   =adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test    =adjusted_R2score_calc(Xtest_in, score_test)
	mrs_train          =mean_squared_error(ridgeRegr.predict(Xtrain_in), ytrain_in)
	mrs_test           =mean_squared_error(ridgeRegr.predict(Xtest_in), ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, ridgeRegr, "Ridge_base" )
	time_end           =time.time() - start_time
	log_record_result("Ridge baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, ridgeRegr, "Ridge_base") #plots residual
	return "Ridge baseline model " ,  str(time_end), str(score_train ) ,str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test)

  

def Laso_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	lasoRegr = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection="cyclic")

	lasoRegr.fit(Xtrain_in, ytrain_in)
	score_test         = lasoRegr.score(Xtest_in, ytest_in)
	score_train        = lasoRegr.score(Xtrain_in, ytrain_in)
	adj_Rscore_train   = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test    = adjusted_R2score_calc(Xtest_in, score_test)
	mrs_train          =mean_squared_error(lasoRegr.predict(Xtrain_in), ytrain_in)
	mrs_test           =mean_squared_error(lasoRegr.predict(Xtest_in), ytest_in)           
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, lasoRegr, "Lasso_base")
	time_end           =time.time() - start_time
	log_record_result("Lasso baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, lasoRegr, "Lasso_base") #plots residual
	return "Lasso baseline model",  str(time_end), str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test)


def Linear_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	linearReg            = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)      
	linearReg.fit(Xtrain_in, ytrain_in)
	score_test           = linearReg.score(Xtest_in, ytest_in)
	score_train          =linearReg.score(Xtrain_in, ytrain_in)
	adj_Rscore_train     = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test      = adjusted_R2score_calc(Xtest_in, score_test)
	mrs_train            =mean_squared_error(linearReg.predict(Xtrain_in), ytrain_in)
	mrs_test             =mean_squared_error(linearReg.predict(Xtest_in), ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, linearReg, "Linear_base")
	time_end             =time.time() - start_time
	log_record_result("Linear baseline regression", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, linearReg, "Linear_base") #plots residual
	return "Linear reg baseline model", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test)
    

def enet_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	eNetregr               = ElasticNet(random_state=0)
	eNetregr.fit(Xtrain_in, ytrain_in)
	score_test            = eNetregr.score(Xtest_in, ytest_in)
	score_train           = eNetregr.score(Xtrain_in, ytrain_in)
	adj_Rscore_train      = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test       = adjusted_R2score_calc(Xtest_in, score_test)
	mrs_train             = mean_squared_error(eNetregr.predict(Xtrain_in), ytrain_in)
	mrs_test              = mean_squared_error(eNetregr.predict(Xtest_in), ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, eNetregr, "Enet_base")
	time_end              =time.time() - start_time
	log_record_result("Enet baseline", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, eNetregr, "Enet_base" ) #plots residual
	return "Enet baseline model", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test)
    



def BayesRidge_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	BayesRidg_reg          = BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=10)

	BayesRidg_reg.fit(Xtrain_in, ytrain_in)

	score_test             = BayesRidg_reg.score(Xtest_in, ytest_in)
	score_train            = BayesRidg_reg.score(Xtrain_in, ytrain_in)
	adj_Rscore_train       = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test        = adjusted_R2score_calc(Xtest_in, score_test)
	mrs_train              = mean_squared_error(BayesRidg_reg.predict(Xtrain_in), ytrain_in)
	mrs_test               = mean_squared_error(BayesRidg_reg.predict(Xtest_in), ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, BayesRidg_reg,"Bayes_base")
	time_end               =time.time() - start_time
	log_record_result("Bayes regression score", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
	plot_residuals(Xtest_in, ytest_in, BayesRidg_reg,"Bayes_base") #plots residual
	return "Bayes regression baseline model ", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test)

#~~~~~~~~~~~~~~~~~~~~~~gridsearch~~~~~~~~~~~~~~~~~~~~~~~~~~~
def linear_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None): 
	start_time = time.time()
	linearRegr         = LinearRegression()
	parameters         = {'fit_intercept':[True,False]}#, 'normalize':[True,False], 'copy_X':[True, False]}
	grid               = GridSearchCV(linearRegr,parameters, cv=2, verbose=5)
	grid.fit(Xtrain_in, ytrain_in)
	y_test_pred        =grid.predict(Xtest_in)
	y_train_pred       =grid.predict(Xtrain_in)
	score_test         =r2_score(ytest_in,y_test_pred)
	score_train        =r2_score(ytrain_in,y_train_pred)
	adj_Rscore_train   =adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test    =adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters    =grid.best_params_
	mrs_train          = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test           = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "Linear_grid")
	time_end           =time.time() - start_time
	log_record_result("Linear Grid search", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "Linear_grid") #plots residual
	return "Linear Grid search ", str(time_end),   str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)

    
def ridge_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	ridgeRegr            = Ridge()
	parameters           = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}
	grid                 = GridSearchCV(ridgeRegr,parameters, cv=5, verbose=10)
	grid.fit(Xtrain_in, ytrain_in)
	y_test_pred          =grid.predict(Xtest_in)
	y_train_pred         =grid.predict(Xtrain_in)
	score_test           =r2_score(ytest_in,y_test_pred)
	score_train          =r2_score(ytrain_in,y_train_pred)
	adj_Rscore_train     = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test      = adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters      =grid.best_params_
	mrs_train            = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test             = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "Ridge_grid")
	time_end             =time.time() - start_time
	log_record_result("Ridge Grid search", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test , best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "Ridge_grid") #plots residual
	return "Ridge Grid search", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)

    
def lasso_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	lassoRegr            = Lasso()
	parameters           = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}
	grid                 = GridSearchCV(lassoRegr,parameters, cv=5, verbose=10)
	grid.fit(Xtrain_in, ytrain_in)
	y_test_pred            =grid.predict(Xtest_in)
	y_train_pred           =grid.predict(Xtrain_in)
	score_test             =r2_score(ytest_in,y_test_pred)
	score_train            =r2_score(ytrain_in,y_train_pred)
	adj_Rscore_train       = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test        = adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters        =grid.best_params_
	mrs_train              = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test               = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "Lasso_grid")
	time_end               =time.time() - start_time
	log_record_result("Lasso Grid search", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "Lasso_grid") #plots residual
	return "Lasso Grid search", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)


def elasticNet_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	eNetreg             = ElasticNet()
	parameters          = {"alpha": [0.001, 0.1, 10]}
	grid                = GridSearchCV(eNetreg,parameters, cv=5, verbose=10)
	grid.fit(Xtrain_in, ytrain_in)
	y_test_pred         = grid.predict(Xtest_in)
	y_train_pred        = grid.predict(Xtrain_in)
	score_test          = r2_score(ytest_in,y_test_pred)
	score_train         = r2_score(ytrain_in,y_train_pred)
	adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters     = grid.best_params_
	mrs_train           = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test            = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "eNet_grid")
	time_end            =time.time() - start_time
	log_record_result("eNet Grid search", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "eNet_grid") #plots residual
	return "eNet Grid search ", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)

def BayesRidge_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
	start_time = time.time()
	BayesRidg_reg        = BayesianRidge()
	parameters           = {"alpha_1": [ 0.001, 0.1]}#, "alpha_2": [0.00001, 0.001, 0.1]}
	grid                 = GridSearchCV(BayesRidg_reg, parameters,n_jobs = -1, cv = 3, verbose=10)
	grid.fit(Xtrain_in, ytrain_in)

	y_test_pred=grid.score(Xtest_in)
	y_train_pred         =grid.score(Xtrain_in)
	score_test=r2_score(ytest_in,y_test_pred)
	score_train=r2_score(ytrain_in,y_train_pred)

	adj_Rscore_train= adjusted_R2score_calc(Xtrain_in, score_train)
	adj_Rscore_test= adjusted_R2score_calc(Xtest_in, score_test)
	best_parameters=grid.best_params_
	mrs_train              = mean_squared_error(y_train_pred, ytrain_in)
	mrs_test               = mean_squared_error(y_test_pred, ytest_in)
	if lime_flag:		
		lime_explainer(Xtrain_in, df_row, grid, "Bayes_grid")
	time_end=time.time() - start_time
	log_record_result("Bayes Grid search", score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
	plot_residuals(Xtest_in, ytest_in, grid, "Bayes_grid") #plots residual
	return "Bayes Grid search score", str(time_end),  str(score_train ),  str(score_test), str(adj_Rscore_train),  str(adj_Rscore_test), str(best_parameters)

'''







