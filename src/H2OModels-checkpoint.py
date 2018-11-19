""" 
#import h2o
#h2o.init(max_mem_size = "2G")             #specify max number of bytes. uses all cores by default.
#h2o.remove_all()                          #clean slate, in case cluster was already running
https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/glm/glm_h2oworld_demo.py
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
help(H2OGeneralizedLinearEstimator)
help(h2o.import_file)

import pandas as pd
import numpy as np

df_df = h2o.import_file(os.path.realpath("/datascience/home/atekola/input/c360_customeradt_lexussegmentation_2017_03_31.csv"))

train, valid, test = covtype_df.split_frame([0.7, 0.15], seed=1234)

glm_multi_v1 = H2OGeneralizedLinearEstimator( model_id='glm_v1',            #allows us to easily locate this model in Flow
                    family='multinomial',
                    solver='L_BFGS')
"""