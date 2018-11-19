import matplotlib
matplotlib.use('Agg')
from directoryPath import mlresult_dir
import datetime
import matplotlib.pylab as plt

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from config_param import project_identifier 

def plot_residuals(X_in, y_in, model, plot_name=" "):
	residual=model.predict(X_in)-y_in
	plt.scatter(y_in,residual,  marker='o')
    #plt.title(plot_name)
	plt.xlabel("True value")
	plt.ylabel("Residual")
	plt.savefig( mlresult_dir + str(project_identifier)+"_plot_residual" + "_" + plot_name + str(datetime.datetime.now().day)+ ':' + str(datetime.datetime.now().month) + ':' + str(datetime.datetime.now().year) + ' ' + str(datetime.datetime.now().hour)+ ':' + str(datetime.datetime.now().minute) + '.' + 'png')



