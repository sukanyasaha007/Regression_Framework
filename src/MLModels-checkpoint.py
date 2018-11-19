import numpy as np
import lightgbm as lgbm
import time
import sys
import warnings
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVR
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV


from log_outputs import log_record_result
from evaluation_metric import adjusted_R2score_calc #custom made
from plot_residual import plot_residuals
from lime_ModelExplainer import lime_explainer
from plot_residual import plot_residuals

# Ignores printing warning 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def ridge_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Ridge Regression Model (or Tikhonov regularization): uses linear least squares loss function and l2-norm regularization
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
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
    return "Ridge baseline model: Time Taken- " ,  str(time_end), "\nScore on Traing Set", str(score_train ), "\nScore on Test Set " ,str(score_test), "\nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), "\nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

  

def laso_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Lasso Regression Model: uses linear least squares loss function and l1-norm regularization
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
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
    return "Lasso baseline model: Time Taken- " ,  str(time_end), "\nScore on Traing Set ", str(score_train ), "\nScore on Test Set " ,str(score_test), "\nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), "\nAdjusted R Square Score on Test Set ",  str(adj_Rscore_test)

def linear_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Linear Regression Model: Ordinary least squares Linear Regression
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
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
    return "Linear reg baseline model: Time Taken- " ,  str(time_end), "\n Score on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

def enet_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function:
    1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    and follows a * L1 + b * L2
    where:
    alpha = a + b and l1_ratio = a / (a + b)
    When parameter l1_ratio = 1 is the lasso penalty. l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha.

    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    eNetregr               = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
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
    return "Enet baseline model: Time Taken- " ,  str(time_end), " Score on Traing Set", str(score_train ), " Score on Test Set " ,str(score_test), " Adjusted R Square Score on Training Set ", str(adj_Rscore_train), " Adjusted R Square Score on Test Set",  str(adj_Rscore_test)


def bayes_ridge_baseline_model(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Bayesian ridge regression : Fit a Bayesian ridge model and optimize the regularization parameters lambda (precision of the weights) and alpha (precision of the noise).
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
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
        lime_explainer(Xtrain_in, df_row, BayesRidg_reg," Bayes_base ")
    time_end               =time.time() - start_time
    log_record_result("Bayes regression score", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, BayesRidg_reg, "Bayes_base") #plots residual
    return "Bayes regression baseline model : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)


#decisiontree baseline model
def mlmodel_decisionTree_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Decision Tree Regressor 
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time          = time.time()
    decisionRegr        = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)     
    decisionRegr.fit(Xtrain_in, ytrain_in)
    score_test          = decisionRegr.score(Xtest_in, ytest_in)
    score_train         =decisionRegr.score(Xtrain_in, ytrain_in)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train           =mean_squared_error(decisionRegr.predict(Xtrain_in), ytrain_in)
    mrs_test           =mean_squared_error(decisionRegr.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, decisionRegr, "Decision_base")                                                
    time_end=time.time() - start_time
    log_record_result("Decision tree regressor baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, decisionRegr, "Decision_base") #plots residual
    return "Decision tree regressor : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

    
#adaboost baseline model   
def mlmodel_adaboostTree_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Decision Tree Regressor 
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time          = time.time()    
    adaboostRegr        = AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss="linear", random_state=None)     
    adaboostRegr.fit(Xtrain_in, ytrain_in)
    score_test          = adaboostRegr.score(Xtest_in, ytest_in)
    score_train         =adaboostRegr.score(Xtrain_in, ytrain_in)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          =mean_squared_error(adaboostRegr.predict(Xtrain_in), ytrain_in)
    mrs_test           =mean_squared_error(adaboostRegr.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, adaboostRegr, "Adaboost_base")
    time_end             =time.time() - start_time
    log_record_result("Adaboost regressor baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, adaboostRegr, "Adaboost_base") #plots residual
    return "Adaboost regressor: \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

#randomforest baseline model
def mlmodel_randomForest_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Random Forest Regressor 
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time           = time.time()
    randomreg = RandomForestRegressor(n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=0, verbose=0, warm_start=False)
    randomreg.fit(Xtrain_in, ytrain_in)
    score_test          = randomreg.score(Xtest_in, ytest_in)
    score_train         =randomreg.score(Xtrain_in, ytrain_in)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          =mean_squared_error(randomreg.predict(Xtrain_in), ytrain_in)
    mrs_test           =mean_squared_error(randomreg.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, randomreg, "Randfor_base")
    time_end            =time.time() - start_time
    log_record_result("Randomforest regressor baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, randomreg, "Randfor_base") #plots residual
    return "Random forest regressor : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

#SVR baseline model 
def mlmodel_svr_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Support Vector Regressor 
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time          = time.time()

    SVRreg              = SVR(kernel="rbf", degree=3, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=True, max_iter=-1)

    SVRreg.fit(Xtrain_in, ytrain_in)
    score_test          = SVRreg.score(Xtest_in, ytest_in)
    score_train         =SVRreg.score(Xtrain_in, ytrain_in)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          =mean_squared_error(SVRreg.predict(Xtrain_in), ytrain_in)
    mrs_test           =mean_squared_error(SVRreg.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, SVRreg, "SVR_base")
    time_end            =time.time() - start_time
    log_record_result("SVR regressor baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, SVRreg, "SVR_base") #plots residual
    return "Support vector regressor \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

#XGB baseline model
def mlmodel_xgb_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Extreme Gradient Boosting Regressor 
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time         = time.time()
    xgbreg             = XGBRegressor()
    xgbreg.fit(Xtrain_in, ytrain_in)
    y_test_pred         =xgbreg.predict(Xtest_in)
    y_train_pred        =xgbreg.predict(Xtrain_in)

    score_test          =r2_score(ytest_in,y_test_pred)
    score_train         =r2_score(ytrain_in,y_train_pred)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          =mean_squared_error(xgbreg.predict(Xtrain_in), ytrain_in)
    mrs_test           =mean_squared_error(xgbreg.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, xgbreg, "XGB_base")
    time_end            =time.time() - start_time
    log_record_result("XGB regressor score", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, xgbreg, "XGB_base") #plots residual
    return "XGB regressor: \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

#LGBM baseline model
def mlmodel_lgbm_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None):
    '''
    Light GBM: Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()

    # LGBM configuration
    lgbm_param = {
    "num_boost_round":25,
    "max_depth" : 3,
    "num_leaves" : 31,
    'learning_rate' : 0.1,
    'boosting_type' : 'gbdt',
    'objective' : 'regression_l2',
    "early_stopping_rounds": None,
    "min_child_weight": 1e-3, 
    "min_child_samples": 20}

    # Calling Regressor using scikit-learn API 
    lgbm_reg = lgbm.sklearn.LGBMRegressor(
    num_leaves=lgbm_param["num_leaves"], 
    n_estimators=lgbm_param["num_boost_round"], 
    max_depth=lgbm_param["max_depth"],
    learning_rate=lgbm_param["learning_rate"],
    objective=lgbm_param["objective"],
    min_sum_hessian_in_leaf=lgbm_param["min_child_weight"],
    min_data_in_leaf=lgbm_param["min_child_samples"])
    lgbm_reg.fit(Xtrain_in, ytrain_in)
    score_test          = lgbm_reg.score(Xtest_in, ytest_in)
    score_train         = lgbm_reg.score(Xtrain_in, ytrain_in)
    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          = mean_squared_error(lgbm_reg.predict(Xtrain_in), ytrain_in)
    mrs_test           = mean_squared_error(lgbm_reg.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, lgbm_reg, "lgbm_base")
    time_end=time.time() - start_time
    log_record_result("lgbm regressor baseline model", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, lgbm_reg, "lgbm_base") #plots residual
    return "LGBM regressor: \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test)

#Catboost baseline model
def mlmodel_catboost_regressor(Xtrain_in, ytrain_in, Xtest_in, ytest_in):
    '''
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time          = time.time()
    catb_reg            = CatBoostRegressor(iterations=2, learning_rate=1, depth=2)
    catb_reg.fit(Xtrain_in, ytrain_in, use_best_model=False, verbose=True)
    #Note if use_best_model=True, a test set should be provided in the fitting function
    y_test_pred         =catb_reg.predict(Xtest_in)
    y_train_pred        =catb_reg.predict(Xtrain_in)
    score_test          =r2_score(ytest_in,y_test_pred)
    score_train         =r2_score(ytrain_in,y_train_pred)

    adj_Rscore_train    = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test     = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train          = mean_squared_error(y_train_pred, ytrain_in)
    mrs_test           = mean_squared_error(y_test_pred, ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, catb_reg,"Cboost_base")
    time_end            =time.time() - start_time
    log_record_result("Cat boost regressor", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train, mrs_test)
    plot_residuals(Xtest_in, ytest_in, catb_reg, "Cboost_base") #plots residual
    return "Cat boost regressor: \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) 

#~~~~~~~~~~~~~~~~~~~~~~gridsearch~~~~~~~~~~~~~~~~~~~~~~~~~~~
def linear_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters = {'fit_intercept':[True,False]}): 
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    linearRegr         = LinearRegression()
    #parameters         = {'fit_intercept':[True,False]}#, 'normalize':[True,False], 'copy_X':[True, False]}
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
    return "Linear Grid search : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)


def ridge_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    ridgeRegr            = Ridge()
    #parameters           = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}
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
    return "Ridge Grid search : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

    
def lasso_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    lassoRegr            = Lasso()
    #parameters           = {'alpha':[0.1, 1.0, 10.0],'fit_intercept':[True,False], 'normalize':[True,False]}
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
    return "Lasso Grid search : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)


def elasticNet_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters = {"alpha": [0.001, 0.1, 10]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    eNetreg             = ElasticNet()
    #parameters          = {"alpha": [0.001, 0.1, 10]}
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
    return "eNet Grid search  : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

def BayesRidge_grid_search(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters = {"alpha_1": [ 0.001, 0.1]}):
    '''
    Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    BayesRidg_reg        = BayesianRidge()
    #parameters           = {"alpha_1": [ 0.001, 0.1]}#, "alpha_2": [0.00001, 0.001, 0.1]}
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
    return "Bayes Grid search score : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#Random forest gridsearch
def randonForest_regressor_gridsearchcv(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, param_grid= { "n_estimators": [10,20,30],"max_features" : ["auto", "sqrt", "log2"],    "max_depth": [1,3,5], "min_samples_split" : [2,4,8], "bootstrap": [True, False], "min_samples_split": [2, 5, 10] }):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time            = time.time()
    randomregrid          = RandomForestRegressor(random_state=0)
    '''param_grid            = { 
    "n_estimators"      : [10,20,30],
    "max_features"      : ["auto", "sqrt", "log2"],
    "max_depth": [1,3,5],
    "min_samples_split" : [2,4,8],
    "bootstrap": [True, False],
    "min_samples_split": [2, 5, 10]
    }
    '''

    grid                   = GridSearchCV(randomregrid, param_grid, n_jobs=-1, cv=5, verbose=5)
    grid.fit(Xtrain_in, ytrain_in)
    score_test             = grid.score(Xtest_in, ytest_in)
    score_train            =grid.score(Xtrain_in, ytrain_in)
    adj_Rscore_train       = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test        = adjusted_R2score_calc(Xtest_in, score_test)
    best_parameters        =grid.best_params_
    mrs_train              = mean_squared_error(grid.predict(Xtrain_in), ytrain_in)
    mrs_test               = mean_squared_error(grid.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "Random_grid")    
    time_end                =time.time() - start_time
    log_record_result("Random reg griadsearchcv ", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "Random_grid") #plots residual
    return "Random reg griadsearchcv  : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#Decision tree gridsearch
def decisionTree_regressor_gridsearchcv(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, param_grid= {'max_depth': [1,2]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time              = time.time()
    decisionreg             = DecisionTreeRegressor(random_state=0)
    #param_grid              = {'max_depth': [1,2]}#, 'max_features': [1, 2]}
    grid = GridSearchCV(decisionreg, param_grid, n_jobs=-1, cv=2, verbose=5)

    grid.fit(Xtrain_in, ytrain_in)
    score_test              = grid.score(Xtest_in, ytest_in)
    score_train             = grid.score(Xtrain_in, ytrain_in)
    adj_Rscore_train        = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test         = adjusted_R2score_calc(Xtest_in, score_test)
    best_parameters         = grid.best_params_
    mrs_train               = mean_squared_error(grid.predict(Xtrain_in ), ytrain_in)
    mrs_test                = mean_squared_error(grid.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "Decision_grid ")
    time_end                =time.time() - start_time
    log_record_result("Decision tree gidsearchcv ", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "Decision_grid ") #plots residual
    return "Decision tree gidsearchcv : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#Adaboost gridsearch
def adaBoost_regressor_gridsearchcv(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, param_grid= {'learning_rate': [.001]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    adaBoostreggrid        = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(random_state=np.random.seed(0)))
    #param_grid             = {'learning_rate': [.001]}
    grid                    = GridSearchCV(adaBoostreggrid, param_grid, n_jobs=None, cv=2, verbose=10)
    grid.fit(Xtrain_in, ytrain_in)
    score_test              = grid.score(Xtest_in, ytest_in)
    score_train             =grid.score(Xtrain_in, ytrain_in)
    adj_Rscore_train        = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test         = adjusted_R2score_calc(Xtest_in, score_test)
    best_parameters         =grid.best_params_
    mrs_train              = mean_squared_error(grid.predict(Xtrain_in), ytrain_in)
    mrs_test               = mean_squared_error(grid.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "AdaBoost_grid  ")
    time_end                =time.time() - start_time
    log_record_result("AdaBoost regressor gidsearchcv  ", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "AdaBoost_grid") #plots residual
    return "AdaBoost regressor gidsearchcv : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#XGB gridsearch
def XGB_regressor_gridsearchcv(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, parameters= {'learning_rate': [.03, 0.05], 'objective':['reg:linear'],'learning_rate': [.03, 0.05, .07],'max_depth': [5, 6, 7]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time              = time.time()
    xgbreg                  = XGBRegressor()
    #parameters              = {'learning_rate': [.03, 0.05], 'objective':['reg:linear'],'learning_rate': [.03, 0.05, .07],'max_depth': [5, 6, 7]}
    grid                    = GridSearchCV(xgbreg,parameters,cv = 5 ,n_jobs = -1, verbose=True)
    grid.fit(Xtrain_in, ytrain_in)
    y_test_pred             =grid.predict(Xtest_in)
    y_train_pred            =grid.predict(Xtrain_in)
    score_test              =r2_score(ytest_in,y_test_pred)
    score_train             =r2_score(ytrain_in,y_train_pred)
    adj_Rscore_train        = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test         = adjusted_R2score_calc(Xtest_in, score_test)
    mrs_train              = mean_squared_error(y_train_pred, ytrain_in)
    mrs_test               = mean_squared_error(y_test_pred, ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "XGB_grid")
    time_end                =time.time() - start_time
    log_record_result("XGB regressor score", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "XGB_grid") #plots residual
    return "XGB gridsearch regressor score : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#LGBM gridsearch
def lgbm_regressor_gridsearchcv( Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, param_grid = {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [20, 40]}) :
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time             = time.time()
    lgbm_reg               = lgbm.LGBMRegressor(num_leaves=31)
    #param_grid             = {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [20, 40]}
    grid = GridSearchCV(lgbm_reg, param_grid, cv=3, verbose=5)
    grid.fit(Xtrain_in, ytrain_in)
    score_test              = grid.score(Xtest_in, ytest_in)
    score_train             =grid.score(Xtrain_in, ytrain_in)
    adj_Rscore_train        = adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test         = adjusted_R2score_calc(Xtest_in, score_test)
    best_parameters         =grid.best_params_
    mrs_train              = mean_squared_error(grid.predict(Xtrain_in), ytrain_in)
    mrs_test               = mean_squared_error(grid.predict(Xtest_in), ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "LGBM_grid")
    time_end                =time.time() - start_time
    log_record_result("LGBM gidsearchcv score", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "LGBM_grid") #plots residual
    return "LGBM gidsearchcv score : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)

#Catboost gridsearch
def catb_regressor_gridsearchcv(Xtrain_in, ytrain_in, Xtest_in, ytest_in, lime_flag=False, df_row=None, params = {'depth': [4, 7, 10]}):
    '''Parameters
    ------
    Xtrain_in: Data Frame of Independent columns for training set
    ytrain_in: Target Vector of training set
    Xtest_in: Data Frame of Independent columns for test set
    ytest_in: Target Vector of test set
    lime_flag: Boolean. Whether to use lime default is False
    df_row=None
    parameter/param_grid: parameter to be tuned using grid search
    Returns Time taken, train score , test score, train ms score, test ms score, adj R score for train, adj R score for test
    '''
    start_time = time.time()
    catb_reg = CatBoostRegressor()
    #params = {'depth': [4, 7, 10]}
    grid = GridSearchCV(catb_reg, params, cv = 3, verbose=True)
    grid.fit(Xtrain_in, ytrain_in)
    y_test_pred=grid.predict(Xtest_in)
    y_train_pred=grid.predict(Xtrain_in)
    score_test=r2_score(ytest_in,y_test_pred)
    score_train=r2_score(ytrain_in,y_train_pred)

    adj_Rscore_train        =adjusted_R2score_calc(Xtrain_in, score_train)
    adj_Rscore_test         =adjusted_R2score_calc(Xtest_in, score_test)
    best_parameters         =grid.best_params_
    mrs_train              = mean_squared_error(y_train_pred, ytrain_in)
    mrs_test               = mean_squared_error(y_test_pred, ytest_in)
    if lime_flag:
        lime_explainer(Xtrain_in, df_row, grid, "Cboost_grid")
    time_end                =time.time() - start_time
    log_record_result("Cat boost regressor gidsearchcv", time_end, score_train, score_test, adj_Rscore_train, adj_Rscore_test, mrs_train=mrs_train, mrs_test=mrs_test, best_param=best_parameters)
    plot_residuals(Xtest_in, ytest_in, grid, "Cboost_grid") #plots residual
    return "Cat boost regressor gidsearchcv : \nTime Taken- " ,  str(time_end), " \nScore on Traing Set", str(score_train ), " \nScore on Test Set " ,str(score_test), " \nAdjusted R Square Score on Training Set ", str(adj_Rscore_train), " \nAdjusted R Square Score on Test Set",  str(adj_Rscore_test) , "\n Best Parameters", str(best_parameters)
