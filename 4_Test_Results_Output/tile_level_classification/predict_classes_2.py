import numpy as np
import pandas as pd
import random
import math
import os
import re
import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import imread
import sys
import keras
from keras.models import load_model	

def main():
	print('starting...')
	testPath = 'G:\\data\\test\\'
	model_dir = 'D:\\models'
	model_files = [f for f in os.listdir(model_dir) if re.match(r'[0-9]+.*\.h5', f)]
	#model_files = model_files[(len(model_files)-1)] #test last model alphabetically / most recent
	if isinstance(model_files, list):
		end_range = len(model_files)-1#'27_class_cnn_color_1'
	else:
		end_range = 1


	for model_counter in range(0, end_range):
		if isinstance(model_files, list):
			model_name = model_files[model_counter]#'27_class_cnn_color_1'
		else:
			model_name = model_files
		print('loading', model_name)
		model = load_model('D:\\models\\'+model_name)
		print('loaded ', model_name)
		save_dir = 'D:\\results\\'+model_name+'\\'
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for counter in range(0, 61):
			print(model_name, counter)
			csv_name = testPath+'test_imgs_'+str(counter)+'.csv'
			imgs_name = testPath+'test_imgs_'+str(counter)+'.npy'
			filenames = pd.read_csv(csv_name)
			filenames = filenames['filename']
			images = np.load(imgs_name)
			results = model.predict(images)
			results2 = pd.DataFrame(results)
			results2.insert(value=filenames, column='filenames', loc=0)
			results2.to_csv(save_dir+'test_imgs_'+str(counter)+'_results.csv')

if __name__ == '__main__':
	main()