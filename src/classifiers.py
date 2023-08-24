# Instituto Balseiro
# Redes Neuronales y Aprendizaje Profundo para Visión Artificial - 2019
# Alumno:  Enrique Nicanor Mariotti
# Carrera: Maestria en Cs de la Ingenieria

import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import time

from sklearn import datasets
from keras.datasets import cifar10, mnist

class LinearClassifier():

	def __init__(self, epochs, learningRate, batchSize, regStrength):
		self.epochs = epochs
		self.learningRate = learningRate
		self.batchSize = batchSize
		self.regStrength = regStrength
		self.w = None

	def train(self, xTrain, yTrain, xTest=None, yTest=None):
		#Dimension
		self.nSamples, self.dim = xTrain.shape[0], xTrain.shape[1]

		#Test data
		if xTest is None:
			xTest = xTrain
			yTest = yTrain

		# Labels and classes
		self.nLabels = np.max(np.unique(yTrain)) + 1

		# Weights
		np.random.seed(1988)
		self.w = 1e-4 * np.random.rand(self.nLabels, self.dim)

		# Format Encoder
		xTrain = np.asarray(xTrain, dtype='float32')
		xTest = np.asarray(xTest, dtype='float32')
		yTrain = np.asarray(yTrain, dtype='int32')
		yTest = np.asarray(yTest, dtype='int32')
		yTrainEncoded = self.__encoder__(yTrain)
		yTestEncoded = self.__encoder__(yTest)

		# Testn and Train History Log
		history = {}
		# Train stats
		history['train_loss'] = np.zeros(self.epochs)
		history['train_acc'] = np.zeros(self.epochs)
		# Test stats
		history['test_loss'] = np.zeros(self.epochs)
		history['test_acc'] = np.zeros(self.epochs)

		for e in range(self.epochs):
			# Stats over training batch
			history['train_loss'][e], _ = self.lossFunction(xTrain, yTrainEncoded) 		
			history['train_acc'][e] = self.meanAccuracy(xTrain, yTrain)

			# Stats over test batch
			history['test_loss'][e], _ = self.lossFunction(xTest, yTestEncoded) 
			history['test_acc'][e] = self.meanAccuracy(xTest, yTest)

			# Print stats
			print('\n#Epoch: {:d}'.format(e))
			print('Train Loss: {:.5f}'.format(history['train_loss'][e]))
			print('Train Accuracy: {:.5f}'.format(history['train_acc'][e]))
			print('Test Loss: {:.5f}'.format(history['test_loss'][e]))
			print('Test Accuracy: {:.5f}'.format(history['test_acc'][e]))
			
			# Stochastic Gradient Descent
			self.__SGD__(xTrain, yTrainEncoded)
		return history

	def __SGD__(self, x, y):
		# Shuffle data
		randomIndex = random.sample(range(self.nSamples), self.nSamples)
		x = x[randomIndex,:]
		y = y[:,randomIndex]
		# Gradient Descent with mini-batches
		for i in range(0, self.nSamples, self.batchSize):
			xBatch = x[i:self.batchSize+i,:]
			yBatch = y[:,i:self.batchSize+i]
			loss, dw = self.lossFunction(xBatch, yBatch)
			regGrad = self.regStrength*self.w
			self.w -= self.learningRate * (dw + regGrad)

	def __encoder__(self, y):
		yEncoded = np.eye(self.nLabels)[y]
		yEncoded = yEncoded.T  
		return yEncoded

	def lossFunction():
		pass

	def meanAccuracy(self, x, y):
		yPredicted = self.predict(x)
		return np.mean(np.equal(y, yPredicted))

	def predict(self, x):
		predictedClass = np.argmax(np.dot(self.w, x.T), axis=0)
		return predictedClass

class Softmax(LinearClassifier):

	def softmaxFunction(self, scores):
		scores -= np.max(scores) # Normalization
		softmax = (np.exp(scores) / np.sum(np.exp(scores), axis=0))
		return softmax

	def lossFunction(self, x, y):
		# Loss
		scores = np.dot(self.w, x.T)
		softmax = self.softmaxFunction(scores)
		lossValue = - np.log(softmax[y==1])
		regLoss = 0.5 * self.regStrength * np.sum(self.w*self.w)
		totalLoss = (np.sum(lossValue) / self.nSamples) + regLoss
		# Grad: (dim,nsamples)(nsamples,nlabels) = (dim,nlabels) -> (nlabels,dim)
		grad = ((1 / self.nSamples) * np.dot(x.T, (softmax - y).T)).T
		return totalLoss, grad

class SVM(LinearClassifier):

	def lossFunction(self, x, y):
		# Loss
		delta = 1
		scores = np.dot(self.w, x.T)
		correctClassScore = scores[y==1]
		margins = np.maximum(0, scores - correctClassScore[np.newaxis,:] + delta)
		margins[y==1] = 0
		lossValue = np.sum(margins)/self.nSamples #Double-axis sum
		regLoss = 0.5 * self.regStrength * np.sum(self.w*self.w)
		totalLoss = lossValue + regLoss
		#Grad
		binaryMask = np.zeros(margins.shape)
		binaryMask[margins > 0] = 1
		count = np.sum(binaryMask,axis=0)
		binaryMask[y==1] = -count
		grad = ((1 / self.nSamples) * np.dot(binaryMask,x))
		return totalLoss, grad

if __name__ == '__main__':

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='method', help='softmax (Softmax Linear Classifier), svm (Support Vector Machine)')

	# Softmax
	softmax = subparsers.add_parser('softmax', help='Softmax Linear Classifier')
	softmax.add_argument('-e', '--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
	softmax.add_argument('-lr', '--learningRate', dest='learningRate', default=0.07, type=float, help='Learning rate')
	softmax.add_argument('-bs', '--batchSize', dest='batchSize', default=16, type=int, help='Number of sample in mini-batches')
	softmax.add_argument('-r', '--regStrength', dest='regStrength', default=0.01,type=float, help='L2 Regularization Strength')
	softmax.add_argument('-d', '--dataType', dest='dataType', default='blobs', type=str, choices={'blobs','cifar10','mnist'}, help='Data type')

	# SVM
	svm = subparsers.add_parser('svm', help='Support Vector Machine')
	svm.add_argument('-e', '--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
	svm.add_argument('-lr', '--learningRate', dest='learningRate', default=0.09, type=float, help='Learning rate')
	svm.add_argument('-bs', '--batchSize', dest='batchSize', default=16, type=int, help='Number of sample in mini-batches')
	svm.add_argument('-r', '--regStrength', dest='regStrength', default=0.01,type=float, help='L2 Regularization Strength')
	svm.add_argument('-d', '--dataType', dest='dataType', default='blobs', type=str, choices={'blobs','cifar10','mnist'}, help='Data type')

	# Dictionary of parsers
	args = parser.parse_args()

	# Info printing
	print('Method: {} | Epochs: {} | Learning Rate: {} | Batch Size: {} | Regularization Strength: {} | Data Type: {}'.format(
		args.method,
		args.epochs,
		args.learningRate,
		args.batchSize,
		args.regStrength,
		args.dataType,
		))

	# Data
	if args.dataType=='blobs':
		xData, yData = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, center_box=(0, 10))
		# Split dataset
		nTrain = int(round(len(yData)*0.8))
		xTrain = xData[:nTrain]
		yTrain = yData[:nTrain]
		xTest = xData[nTrain:]
		yTest = yData[nTrain:]

	if args.dataType=='cifar10':
		(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
		# Flatten
		xTrain = xTrain.reshape(xTrain.shape[0],np.prod(xTrain.shape[1:])) 
		xTest = xTest.reshape(xTest.shape[0],np.prod(xTest.shape[1:])) 
		yTrain = yTrain.reshape(-1)
		yTest = yTest.reshape(-1)
		print('xTrain shape:', xTrain.shape)
		print('yTrain shape:', yTrain.shape)

	if args.dataType=='mnist':
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
		# Flatten
		xTrain = xTrain.reshape(xTrain.shape[0],np.prod(xTrain.shape[1:])) 
		xTest = xTest.reshape(xTest.shape[0],np.prod(xTest.shape[1:])) 
		yTrain = yTrain.reshape(-1)
		yTest = yTest.reshape(-1)
		print('xTrain shape:', xTrain.shape)
		print('yTrain shape:', yTrain.shape)

	# Method command
	if args.method=='softmax':
		# Softmax Linear Classifier
		start_time = time.time()
		SM = Softmax(epochs=args.epochs, learningRate=args.learningRate, batchSize=args.batchSize, regStrength=args.regStrength)    		
		history = SM.train(xTrain, yTrain, xTest, yTest)
		end_time = time.time()
		print('\nElapsed KNN Time: {:.5f} seconds'.format(end_time-start_time))

	if args.method=='svm':
		# Support Vector Machine
		start_time = time.time()
		SVM = SVM(epochs=args.epochs, learningRate=args.learningRate, batchSize=args.batchSize, regStrength=args.regStrength)    		
		history = SVM.train(xTrain, yTrain, xTest, yTest)
		end_time = time.time()
		print('\nElapsed KNN Time: {:.5f} seconds'.format(end_time-start_time))
			
	# History plot
	fig_acc = plt.figure(1)
	plt.plot(history['train_acc'], label='train_acc')
	plt.plot(history['test_acc'], label='test_acc')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	fig_acc.savefig('acc_{}_{}.png'.format(args.dataType,args.method), bbox_inches='tight')

	fig_loss = plt.figure(2)
	plt.plot(history['train_loss'], label='train_loss')
	plt.plot(history['test_loss'], label='test_loss')
	plt.xlabel('Number of epochs')
	plt.ylabel('Loss')
	plt.legend()
	fig_loss.savefig('loss_{}_{}.png'.format(args.dataType,args.method), bbox_inches='tight')