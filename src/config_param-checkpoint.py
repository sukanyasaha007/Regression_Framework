#Config file 

split_frac                           =0.2    
#test_size                           =0.15
#validation_size                     =0.30 

null_fraction                        =0.01  #fraction above which all columns with null values rejeced
lower_bound                          =10    #lower bound unique values
upper_bound                          =1252255 #upper bound of unique values

#csv format data
raw_data_test_csv      ="c360_customeradt_lexussegmentation_2014_03_31.csv"
raw_data_train_val_csv ="c360_customeradt_lexussegmentation_2014_03_31.csv"

#h5 format data converted from the above files
raw_data_train_valh5 = "train_val.h5"
raw_data_testh5 = "test.h5"

project_identifier = "msrp"
