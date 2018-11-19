import os
import sys
import time
import datetime
import lime
import lime.lime_tabular
from directoryPath import mlresult_dir
from config_param import project_identifier 
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def lime_explainer(df_Xtrain, df_row, model_predictor, alogorithm_name=" "):
	#df_Xtrain is the training features excluding target
	#df_row is observation to be explained. It is 1D array
	#model_predictor is the model dependent regresser
	try:
		explainer  = lime.lime_tabular.LimeTabularExplainer(df_Xtrain,verbose=True, mode='regression')
	except:
		explainer  = lime.lime_tabular.LimeTabularExplainer(df_Xtrain.values ,verbose=True, mode='regression')
	exp        = explainer.explain_instance(df_row.values, model_predictor.predict)
	#exp.show_in_notebook(show_table=True)
	exp_result = exp.as_list()

	f = open(mlresult_dir + str(project_identifier) +"_log_lime.txt","a")
	f.write(sttime + '\n')
	f.write(str(alogorithm_name)+"\t"+str(exp_result)+ "\n")
	f.close()

