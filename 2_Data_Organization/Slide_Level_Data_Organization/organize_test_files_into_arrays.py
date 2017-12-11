from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import math
import os
import sklearn
from sklearn.metrics import auc, roc_curve
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import imread
import joblib
import sys
sys.setrecursionlimit(10000)

def load_image(filename, path):
	filename = path+filename
	image = imread(filename)
	#print(image.shape[1])
	if (len(image.shape) > 2):
		if(image.shape[1]==150):
			print(filename)
			image = np.rollaxis(image, 2)
			image = image[np.newaxis]
	else:
		image = np.zeros((1, 3, 150, 150))
		filename = '0'
	return image, filename

def main():
	print('starting...')
	testPath = 'E:\\brca_thumbnails\\all\\'
	allFiles=pd.DataFrame(columns=['filenames'])
	num_imgs_per_array = 5000
	#	model = load_model('D:\\models\\27_class_cnn_color_1.h5')

	test_files = [f for f in os.listdir(testPath) if f.endswith('.jpg')]
	#test_files = pd.read_csv('D:\\brca_thumbnails\\data\\test_all\\thumbnail_data.csv')
	#test_files = test_files['full_path']
	random.seed(1149)
	#shuffled_indices = np.random.permutation(np.arange(len(test_files)))
	#test_files = test_files[shuffled_indices]

	num_files = len(test_files)
	num_loops = math.ceil(num_files/num_imgs_per_array)

	for counter in range(0, num_loops):
		#print(counter)
		start_idx = counter*num_imgs_per_array
		if counter != (num_loops-1):
			end_idx = (counter + 1)*num_imgs_per_array
		else:
			end_idx = num_files
		
		print(start_idx, end_idx)

		current_files = test_files[start_idx:end_idx]

		temp_array = joblib.Parallel(n_jobs=10)(joblib.delayed(load_image)(file, testPath) for file in current_files)
		images, filenames = zip(*temp_array)
		images = np.array(images)
		images = images[:,0,:,:,:]
		#allFiles.insert(value=filenames, column='filenames_'+str(counter), loc=len(allFiles.columns))
		#print(filenames)
		files1 = pd.DataFrame(list(filenames), columns=['filename'])
		files1.to_csv(testPath+'test_imgs_'+str(counter)+'.csv')
		np.save(file=testPath+'test_imgs_'+str(counter)+'.npy', arr=images)
		print('saved ', 'test_imgs_'+str(counter))

	#allFiles.to_csv(testPath+'test_files.csv')
	print('Complete.')

if __name__ == '__main__':
	main()