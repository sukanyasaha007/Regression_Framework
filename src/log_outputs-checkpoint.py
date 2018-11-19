import os
import sys
import time
import datetime
from directoryPath import parent_dir, data_dir, parent_dir_project, mlresult_dir
from config_param import project_identifier 


#print (mlresult_dir+ "log_result.txt")
ts = time.time()
sttime = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H:%M:%S - ')

def log_record_result( alogorithm_name, time_taken, r2score_train, r2score_test, adjr2score_train,adjr2score_test, mrs_train=None, mrs_test=None, best_param=None):
     
    f = open(mlresult_dir + str(project_identifier)+ "_log_result.txt","a")
    f.write(sttime + '\n')
    f.write(str(alogorithm_name)+"\t"+"total time"+"\t"+ str(time_taken) +"\t" +"score:"+ "\t\t" +"R2 train score =" + "\t" +str(r2score_train)+"\t"+"R2 test score =" +"\t" +str(r2score_test)+"\t"+"adjusted R2 train score ="  +"\t" + str(adjr2score_train)+"\t" +"adjusted R2 test score ="+ "\t" +str(adjr2score_test) +"\t" +"mrs train score ="+"\t"+str(mrs_train)+"\t"+"mrs test score ="+"\t"+str(mrs_test)+"\t"+"best param ="+"\t"+str(best_param)+"\n")
    f.close()
        
