# Instituto Balseiro
# Redes Neuronales y Aprendizaje Profundo para Visión Artificial - 2019
# Alumno:  Enrique Nicanor Mariotti
# Carrera: Maestria en Cs de la Ingenieria

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter as counter
import argparse
import time

from sklearn import datasets
from keras.datasets import cifar10, mnist

class KNN():

	def __init__(self, k, classes):
		self.k = k # k-nearest neighbors
		self.classes = classes # k-nearest neighbors

	def fit(self,xTest,xData,yData):
 		# K Sorted nearest distances & indexes
		xData = np.asarray(xData, dtype='float32')
		yData = np.asarray(yData, dtype='float32')
		xTest = np.asarray(xTest, dtype='float32')

		prediction = np.zeros(xTest.shape[0], np.int32)
		for idx,dataPoint in enumerate(xTest):
			distance = dataPoint - xData
			neighbor_distance_index = np.linalg.norm(distance, ord=2, axis=-1)
			neighbor_distance_index = np.vstack((neighbor_distance_index,yData)).T
			if self.k==1:
				idmin = np.argmin(neighbor_distance_index[:,0])
				prediction[idx] = yData[idmin] 
			else:
				knn =  neighbor_distance_index[neighbor_distance_index[:,0].argsort()][:self.k]
				prediction[idx] =  self.__counter__(knn)
		return prediction

	def __counter__(self,knn):
		labels = [knn[i,1] for i in range(self.k)]
		prediction = counter(labels).most_common(1)[0][0]
		return prediction

def meanAccuracy(y0, y1):
	return np.mean(np.equal(y0, y1))

if __name__ == "__main__":

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command', help='test / mnist / cifar10')

	# Test
	test = subparsers.add_parser('test', help='2D demostration')
	# Cifar10
	predict = subparsers.add_parser('predict', help='CIFAR10 or MNIST')
	predict.add_argument('-k', '--k', dest='k', default=1, type=int, help='K Nearest Neighbors')
	predict.add_argument('-d', '--dataType', dest='dataType', type=str, choices={'cifar10','mnist'}, required=True, help='Data type')
	
	# Dictionary of parsers
	args = parser.parse_args()

	# Test command
	if args.command=='test':
		# Data settings
		classes = 5		
		xData, yData = datasets.make_blobs(n_samples=200, centers=classes, n_features=2, center_box=(0, 9))

		# Decision boundary
		x_min = 0
		x_max = 10
		y_min = 0
		y_max = 10
		resolution = 0.1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
		xTest = np.vstack((xx.ravel(), yy.ravel())).T
		
		# K loop
		for k in [1,3,7]:
			#Fitting & prediction
			start_time = time.time()
			knn = KNN(k, classes)
			prediction = knn.fit(xTest, xData, yData)
			end_time = time.time()
			print('\nElapsed KNN Time: {} seconds'.format(end_time-start_time))

			# Plot the contour map
			prediction = prediction.reshape(xx.shape)
			fig = plt.figure(k)
			plt.contourf(xx, yy, prediction, cmap='GnBu')
			plt.scatter(xData[:,0],xData[:,1], c=yData)
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('K={} Nearest Neighbors Demo Example'.format(k))
			plt.axis([0, 10, 0, 10])
			fig.savefig('K_{}NN.png'.format(k), bbox_inches='tight')

	if args.command=='predict' and args.dataType=='cifar10':
		# Data
		(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
		
		# Flatten
		xTrain = xTrain.reshape(xTrain.shape[0],np.prod(xTrain.shape[1:])) 
		xTest = xTest.reshape(xTest.shape[0],np.prod(xTest.shape[1:])) 
		yTrain = yTrain.reshape(-1)
		yTest = yTest.reshape(-1)
		print('xTrain shape:', xTrain.shape)
		print('yTrain shape:', yTrain.shape)
		
		# Parameters
		k = args.k
		classes = np.max(np.unique(yTrain)) + 1

		#KNN
		start_time = time.time()
		knn = KNN(k, classes)
		prediction = knn.fit(xTest[:10], xTrain, yTrain)
		end_time = time.time()
		print('\nElapsed KNN Time: {} seconds'.format(end_time-start_time))
		print('Mean Accuracy: {}'.format(meanAccuracy(prediction, yTest[:10])))

	if args.command=='predict' and args.dataType=='mnist':
		# Data
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
		
		# Flatten
		xTrain = xTrain.reshape(xTrain.shape[0],np.prod(xTrain.shape[1:])) 
		xTest = xTest.reshape(xTest.shape[0],np.prod(xTest.shape[1:])) 
		yTrain = yTrain.reshape(-1)
		yTest = yTest.reshape(-1)
		print('xTrain shape:', xTrain.shape)
		print('yTrain shape:', yTrain.shape)
		
		# Parameters
		k = args.k
		classes = np.max(np.unique(yTrain)) + 1

		#KNN
		start_time = time.time()
		knn = KNN(k, classes)
		prediction = knn.fit(xTest[:10], xTrain, yTrain)
		end_time = time.time()
		print('\nElapsed KNN Time: {} seconds'.format(end_time-start_time))
		print('Mean Accuracy: {}'.format(meanAccuracy(prediction, yTest[:10])))