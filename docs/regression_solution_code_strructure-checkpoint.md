
# Code Structure Overview

This folder can be utilized to store all python scripting files from this project. 
All the files should be python scripts only (with .py extension)
The contents of this folder should be checked in to bitbucket.

The entire pipeline has three main components. 
   1. The actual model related codes which is divided in to three groups by itself (Machine learning models, statistical models and deep learning models)
   2. utility codes (includes reading_data, plot, variable selection, model, parameter configuration, etc) 
   3. and finally the part in which everything integrated to work together (msrp_regression.py file).
   
Below, brief description is given about each file.

## 1. data_prep.py  
This file contains some functions that are used to actually select columns used to select features. This includes: 
         1.1 Removing columns whose null values are above 10% and after that it removes the remaining Null values based on rows (Any row with any number of null gets removed)
         1.2 It then compares train and test samples and filters them with common columns
         1.3 The sum of three columns is added as one label column andremovd the individual columns
         1.4 Finally the data is saved with .h5 extension
       
## 2.  data_summary.py 
This file contains a function that gives data summary such as mean, max, min, number of missing values ets of each column. It also produces plots of these quantities

## 3. directoryPath.py 
This file represents home, data, project directories as variable

## 4. DLmodels.py
This file contains deep learning models

## 5. load_package.py
This file contains packages to be loaded

6. log_output.py 
This file keeps record of performances measures of each algorithm. It writes its outputs in a file called "log_results.txt" file save it in mlresults directory

7. msrp_regression.py 
In this file all the necessary functions from other files gets called. This includes variable selection,  data_split, modleing, etc. To run this pipeline, one has to just excute "python msrp_regression.py"

8. metric.py
 All the custom created evaluation functions are found in this file
 
9. MLModels.py
 All the machine learning algorithms are here
10. modelData_split.py
This contains a function that takes in a data frame and splits the data into training/test and even to validation set

11. param.py
This file contains list of parameters used in the whole process. The parameters are parameters that are not used more than once place in the process. 

12. plot_residual.py
This file contains a function that plots residual

13. reading_file.py 
This contains a function that reads any kind of file based on the file extension

14. StatisticalModels.py
This contains models that are statistical by their very nature such as linear regression, ridge and lasso etc

15. var_transform.py
This file contains function used for variable transformation etc such as outlier detection.  At the moment it is not being used.

