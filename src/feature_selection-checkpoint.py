import sklearn
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
def feature_ranking(X,y, estimator='lin'):
    '''
    gives feature rankings for all columns
    parameter
    -------
    X: Data frame of Indepenedent columns where categorical columns are encoded
    y: Target Variable
    estimator: Algorithm to measure rank of the features. Options-
    lin = Linear Regression
    lasso= Lasso
    ridge= Ridge
    svm= Support Vector Machine
    dt= Decision Tree Regressor
    rf= Random Forest Regressor
    boost= Ada Boost Regressor
    
    '''
    # Create the RFE object and rank each feature
    
    lasso=sklearn.linear_model.Lasso()
    ridge=sklearn.linear_model.Ridge()
    svm=sklearn.svm.SVR()
    dt=sklearn.tree.DecisionTreeRegressor()
    rf=sklearn.ensemble.RandomForestRegressor()
    boost=sklearn.ensemble.AdaBoostRegressor()
    
    estimator_dict= {'lin': lin, 'ridge': ridge, 'svm': svm, 'dt': dt, 'rf': rf, 'boost':boost} 
    
    rfe = RFE(estimator_dict[estimator])
    rfe.fit(X, y)
    ranking = rfe.ranking_
    rank_df_lasso=pd.DataFrame(ranking, index=X.columns)
    return  rank_df
    
def select_k_best(X_encoded, y, score_func=f_regression, k=400 ):
    '''
    Selects independent columns based on k parameter
    Parameters : 
    X_encoded: Df of independent columns with encoded categorical columns
    score_func: which score function to use. Below are the options
        For regression: f_regression, mutual_info_regression
        For classification: chi2, f_classif, mutual_info_classif
    k : how many featues to keep
    '''
    selectk=SelectKBest(score_func, k )
    selectk.fit(X_encoded, y)
    X_new= pd.DataFrame(selectk.transform(X_encoded))
    X_new.head()
    print(selectk.scores_)