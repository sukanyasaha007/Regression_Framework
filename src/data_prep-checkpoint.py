import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import plotly as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from log_outputs import mlresult_dir
from config_param import project_identifier 



ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def remove_null(df_in, fraction):
    df_1=df_in[df_in.columns[df_in.isnull().mean() < fraction]] #lower limit of missing                                                                                   #values
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a") 
    f.write(sttime + '\n')
    f.write("The original dimension of the data is" +"\t"+ str(df_in.shape) +"\t"+ " and after columns whose missing values are" +"\t"+ str(fraction* 100) +"\t"+ " % and above are removed, the size is " +"\t"+ str(df_1.shape) + "\n" )
    df_out=df_1.dropna()                                        #removing rows with null
    f.close()
    #~~~~~~~~~~~~~~~logging~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir + str(project_identifier) +"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Finally after all rows with one or more missing values are removed , the size is " +"\t"+ str(df_out.shape) + "\n" )
    f.close()
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    

    return df_out

#This function selects variables whose number of unique values are between a certain value and also removes #catagorical variables
def feature_selection(df_in, n1, n2):                   #n1=lower bound for unique               
    #lower bound of 
                                                        # unique elements  values and n2 is higher 
    n_uniq_values=df_in.nunique()                       #number of unique values in column
    n_uniq_boolean= n_uniq_values >= n1                 #boolean values of Columns with                                                                           #unique values greater than n1
    df_in=df_in[n_uniq_boolean.index[n_uniq_boolean]]   #Filtering col with the boolean
    
    #upper bound of                                      # unique elements
    n_uniq_values1=df_in.nunique()                       #unique values in column
    n_uniq_boolean1= n_uniq_values1 <= n2                #Columns with unique values                                                                                #greater less than  n2
    df_in=df_in[n_uniq_boolean1.index[n_uniq_boolean1]]  #creating boolean 
    df_out=df_in._get_numeric_data()                     #removes catagorical variables
    
    #~~~~~~~~~~~~~~~~~Logging~~~~~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir +str(project_identifier) +"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Keeping only features whose number of unique values are between" +"\t"+str(n1)+ "\t" + str(n2)+"."+"\t"+ "The resulting dataframe has a dimension of"+"\t"+ str(df_out.shape) + "\n" )
    f.close()
    return df_out


#Selectiing the feature intersection of two data frames
def selecting_common_col(df_in1, df_in2):
    if len(df_in1.columns) > len(df_in2.columns): #first data
        col2=df_in2.columns
        df_out1=df_in1[col2]
        df_out2=df_in2
        
    elif len(df_in2.columns) > len(df_in1.columns): #second data
        col1=df_in1.columns
        df_out2=df_in2[col1]
        df_out1=df_in1
        
    else:
        df_out1=df_in1
        df_out2=df_in2
    
    #~~~~~~~~~~~~~~~~~Logging~~~~~~~~~~~~~~~~~~~~~~
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("After filtering the frame with more number of columns with columns of the fame with less number of columns, we have" +"\t"+ str(df_in1)+"\t"+ "and" + "\t" + str(df_in2) + "with dimensions of"+ "\t" +str(df_in1.shape)+"\t"+ "and"+ str(df_in2.shape) + "respevtively"+ "\n" )
    f.close()

    return df_out1, df_out2
        
#creating target column by adding three of teh columns and eventually deleting these three columns. The final result is the features and the label are different.

def creating_label_features(df_in):
    df_y =  df_in['dep_new_msrp_lexus_12mo'] + df_in['dep_cpo_msrp_lexus_12mo'] + df_in['dep_used_msrp_lexus_12mo']
    df_X =  df_in.drop(columns=['dep_new_msrp_lexus_12mo','dep_cpo_msrp_lexus_12mo','dep_used_msrp_lexus_12mo']) #the three features to be combined
    f = open(mlresult_dir + str(project_identifier)+"_log_data_preparation.txt","a")
    f.write(sttime + '\n')
    f.write("Finally, after summing three columns to get one dependent variable, we end up with df_X and df_y" +"\t"+ "with dimension" + "\t" + str(df_X.shape) +  "and"+ str(df_y.shape) + "respevtively"+ "\n" )
    f.close()
    return df_X, df_y

    
### Adding New functions 11/13/2018

def null_value_counts(df):
    '''
    Takes a data frame and returns count of null values with respect to each feature
    Parameter
    ---------
    df: Dataframe
    Returns: dataframe containing percentage of Counts of null values of each variable
    '''
    count_percentage = pd.DataFrame(((df.isnull().sum())/df.shape[0])*100, columns= ['Percentage of missing values'])
    
    return count_percentage['Percentage of missing values'].sort_values(ascending=False)
# Get the list of columns having only one unique value:
def get_cols_with_one_unique_value(df):
    '''
    Takes a data frame and returns list of columns with only one unique value
    Parameter
    ---------
    df: Dataframe
    Returns: list containing independent variable names which have only one unique value
    
    '''
    global cols_with_one_unique_val
    cols_with_one_unique_val= []
    for col in df.columns:
        if df[col].nunique()==1:
            cols_with_one_unique_val.append(col)
    return cols_with_one_unique_val

#remove the columns that have only one unique value

def remove_cols_with_one_unique_value(df):
    '''
    Takes a data frame and returns self after removing columns having only one unique value
    Parameter
    ---------
    df: Dataframe
    Returns: Dataframe without columns that have only one unique value
    
    '''
    get_cols_with_one_unique_value(df)
    return df.drop(labels=cols_with_one_unique_val, axis=1)

# Count number of numerical columns
def get_numeric_cols(df):
    '''
    Keeps only the numerical columns
    Parameter: A dataframe
    Returns: Only numeric colums of the dataframe
   
    '''
    return df.select_dtypes(exclude='O')

# Count number of categorical columns
def get_non_numeric_cols(df):
    '''
    Keeps only the numerical columns
    Parameter: A dataframe
    Returns: Only numeric colums of the dataframe
   
    '''
    return df.select_dtypes(include='O')

# Data Type Casting
def type_casting(df, num_level, cat_lavel):
    '''
    Takes a data Frames and drops categorical column which has more than certain number of unique levels and if any numerical column has less than num_level of unique values then the numerical column is converted to categorical column
    Parameters
    -------
    df: A dataframe 
    cat_level: number of levels, if any categorical column has more than this number of unique levels the categorical column is dropped
    num_level: number of unique values a numeric column must have, if any numerical column has less than these much unique values then the numerical column is converted to categorical column
    
    Returns: Imputed dataframe
    '''
    num_to_cat=[]
    count=0
    for col in df.select_dtypes(exclude='object').columns:
        if df[col].nunique()<num_level and not col.startswith('dep'):
            num_to_cat.append(col)
            df[col]= df[col].astype(np.object)
            count+=1
    print('Total number of numerical Variables changed to Categorical Variable is '+ str(count))
    count=0
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique()>cat_lavel and not col.startswith('dep'):
            print('Dropping Categorical Variable ' + str(col))
            df.drop(col, axis=1, inplace=True)
            count+=1
    print('Total number of Categorical Variables dropped is '+ str(count))
    return df, num_to_cat

#Missing value Imputation
def missing_value_imputation(df, method, thresh_cat, thresh_num):
    '''
    Takes a data Frames and replaces the missing values with zero, median, mode
    Parameters
    -------
    df: A dataframe 
    method: the method of imputation
    thresh_cat: threshold value or percentage of null rows in a categorical column beyond which the categorical column is dropped
    
    thresh_num: threshold value or percentage of null rows in a numerical column beyond which the numerical column is dropped
    
    Returns: Imputed dataframe
    '''
    df.dropna(how='all', inplace=True, axis=1) #Drops column with all null values
    rows=df.shape[0]
    thresh_cat= np.round((rows*thresh_cat)/100)
    thresh_num= np.round((rows*thresh_num)/100)
    #Impute categorical columns
    #if the column has more than threshold values missing then remove it
    #if its less than 30% values are missung then replace with most frequesnt class
    #if its more than 30% but less than 50% values are missing then create a new class 'NA'
    #get a dataframe of only categorical variables
    df_cat= get_non_numeric_cols(df)
    df_cat.dropna(axis=1, thresh=thresh_cat, inplace=True)
    for cat_col in df_cat.columns:
        null_counts=df_cat[cat_col].isnull().sum()
        if null_counts>0 and null_counts<rows*thresh_cat :
            df_cat[cat_col].fillna(df_cat[cat_col].value_counts().index[0], inplace=True)
       # else :
            #df_cat[cat_col].fillna('NA', inplace=True)
    
    #Imputation for numeric Columns
    df_num= get_numeric_cols(df)
    #Remove columns where more than 60% values are null
    df_num.dropna(axis=1, thresh=thresh_num, inplace=True)
    #imputation
    fill_NaN = Imputer(missing_values='NaN', strategy=method, axis=0)
    imputed_df = pd.DataFrame(fill_NaN.fit_transform(df_num))
    imputed_df.columns = df_num.columns
    #imputed_df.index = df_num.index
    '''for i in df.select_dtypes(exclude='number').columns:
        imputed_df[i]=df[i]
    '''
    result= pd.concat([df_cat, imputed_df])
    return result

#Outlier Detection
# a number "a" from the vector "x" is an outlier if 
# a > median(x)+1.5*iqr(x) or a < median-1.5*iqr(x)
# iqr: interquantile range = third interquantile - first interquantile

def outlier_treatment_dataframe(df):
    '''
    Takes a data frame a replces outliers with 5 and 95 percentile for each numeric column
    Parameter: Dataframe
    Returns: Same dataframe with outliers replaced by 5 and 95 percentile for each numeric column
    '''
    df_num= get_numeric_cols(df)
    df_non_num= get_non_numeric_cols(df)
    icq=df_num.quantile(.75)-df_num.quantile(.25)
    h=df_num.quantile(.75)+1.5*icq
    l=df_num.quantile(.25)-1.5*icq
    df_num.mask(df_num <l, df_num.quantile(.05), axis=1, inplace=True)
    df_num.mask(df_num >h, df_num.quantile(.95), axis=1, inplace=True)
    result=pd.concat([df_num, df_non_num], sort='True', axis=1)
    return result

def outlier_treatment_vector(x):
    '''
    Takes a pandas series or dataframe column or any other vector and returns its outliers replaced by 5 and 95 percentile for each numeric column
    Parameter: a vector
    Returns: A vector with its outliers replaced by 5 and 95 percentile for each numeric column
   '''
    icq=x.quantile(.75)-x.quantile(.25)
    h=x.quantile(.75)+1.5*icq
    l=x.quantile(.25)-1.5*icq
    x.replace(x[x <l], x.quantile(.05), inplace=True)
    x.replace(x[x >h], x.quantile(.95), inplace=True)
    return x


def label_encode(df):
    #Encoding labels of categorical variables
    '''
    Takes a data frame and returns the same with all the categorical labels encoded in integer 
    Parameter
    -----
    df: a dataframe
    Returns: encoded self
    '''
    le=LabelEncoder() # create instance of sklearn label encoder method
    df_cat=get_non_numeric_cols(df)
    df_cat_encoded=pd.DataFrame(None)
    for i in df_cat.columns:
        encoded=pd.DataFrame(le.fit_transform(df_cat[i]))
        df_cat_encoded= pd.concat((encoded, df_cat_encoded), axis=1)
    df_cat_encoded.columns=df_cat.columns
    df_encoded=pd.concat((df_cat_encoded, df.select_dtypes(include='number')), axis=1)
    return df_encoded


# Correlation 
def remove_col_with_corr(df, correlation_threshold=.5):
    '''
    Takes a data frame and returns the it after removing columns based on correlation threshold
    Parameter:
    -----
    df :a dataframe
    correlation_threshold: threshold value of correlation beyond which features need to be dropped
    Returns: dataframe of features exceeding the correlation threshold
    '''
    global record_collinear
    corr_matrix = df.corr()
    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]
    # Dataframe to hold correlated pairs
    record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])
    # Iterate through the columns to drop to record pairs of correlated features
    for column in to_drop:

    # Find the correlated features
        corr_features=list(upper.index[upper[column].abs() > correlation_threshold])

    # Find the correlated values
        corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
        drop_features = [column for _ in range(len(corr_features))]    

    # Record the information (need a temp df for now)
        temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,'corr_feature': corr_features,'corr_value': corr_values})
    # Add to dataframe
        record_collinear = record_collinear.append(temp_df, ignore_index = True)
    df.drop(labels=record_collinear['drop_feature'], axis=1)
    return record_collinear, df.drop(labels=record_collinear['drop_feature'], axis=1)

# Collinearity using VIF
def remove_collin_with_vif(X, vif_threshold=5):
    '''
    Takes a data frame and returns the columns to remove based on vif threshold
    Parameter:
    -----
    X :a dataframe with encoded categorical columns
    vif_threshold: threshold value of vif beyond which features need to be dropped
    Returns: dataframe of features exceeding the vif threshold
    '''
    
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc) + ' and with vif= ' +                         str(max(vif)))
            variables = np.delete(variables, maxloc)
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[cols[variables]]

#Remove database name from each variable if the variable name and database name are joined by "."
def remove_database_name(df):
    '''
    Takes a DataFrame and returns variable names if the variabel names were joined with "." to the database name
    Parameter: A dataframe
    Returns: List of column names containing only variable names
    '''
    var=[]
    for col in df.columns:
        #if i.startswith('d'):
        var.append(col.split('.')[1])
    return var
#Find dependent variable names that starts with 'dep'
def find_dep_var(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns list of dependent columns
    Parameter: A dataframe
    Returns: List of dependent variables
    '''
    dep_var=[]
    for col in df.columns:
        if col.startswith('dep'):
            dep_var.append(col)
    return dep_var
#Find matrix of independent variable names when dependent variable names starts with 'dep'
def find_indep_feat(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns dataframe of independent columns
    Parameter: A dataframe
    Returns: A dataframe of independent variables
    '''
    indep_var=[]
    for col in df.columns:
        if not col.startswith('dep'):
            indep_var.append(col)
    X=pd.DataFrame(None)
    for i in indep_var:
        X=pd.concat((X, df[i]), axis=1)
    return X
#Find independent variable names when dependent variable names starts with 'dep'
def find_indep_var(df):
    '''
    Takes a data frame where dependent variable names starts with 'dep' and returns list of independent columns' names
    Parameter: A dataframe
    Returns: A list of independent variables
    '''
    indep_var=[]
    for col in df.columns:
        if not col.startswith('dep'):
            indep_var.append(col)
    return indep_var