# Instituto Balseiro
# Redes Neuronales y Aprendizaje Profundo para Visión Artificial - 2019
# Alumno:  Enrique Nicanor Mariotti
# Carrera: Maestria en Cs de la Ingenieria

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

class LinearRegression():

	def __init__(self,x,y):
		self.i = x.shape[0]
		self.n = x.shape[1]-1

	def fit(self,x, y):
		# Ordinary least squares closed solution
		# Moore–Penrose inverse
		mp_inverse = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
		# Moore–Penrose fitted weights		 
		self.fitted_weigths = np.dot(mp_inverse, y) 
		y_fitted = np.dot(x,self.fitted_weigths)
		return y_fitted

def generate_data(i,n):
	np.random.seed(1988)
	# Linear data + Gaussian Noise (arbitrary values)
	true_weigths = np.random.uniform(0, 10, n+1) # Weights & Bias
	noise = np.random.normal(0, 3, i)
	# Generate Data
	x = np.random.uniform(-10,10, (i,n)) 
	x = np.hstack((x, np.ones((i,1)))) # Add Bias
	y = np.dot(x,true_weigths) + noise
	return x,y 

if __name__ == "__main__":

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command', help='test (for 1D demostration) or fit (for N-D fitting)')

	# Fit: n_points & n_dim parsers
	fit = subparsers.add_parser('fit', help='N-D fitting')
	fit.add_argument('-i', type=int, metavar='points_number', help='points_number', default=1000)
	fit.add_argument('-n', type=int, metavar='n-dimension', help='n-dimension', default=5)

	# Test: n_points=100 & n_dim=1
	test = subparsers.add_parser('test', help='1D demostration')

	# Dicttionary of parsers
	args = parser.parse_args()

	# Test command
	if args.command=='test':
		# Data settings
		n = 1 #1-dimensional data
		i = 100 #100 points
		x,y = generate_data(i,n)
		# Object
		start_time = time.time()
		LR = LinearRegression(x,y)
		y_fitted = LR.fit(x,y)
		end_time = time.time()
		# Fitted weights
		print('Fitted weights vector: {}'.format(LR.fitted_weigths))
		print('Elapsed Linear Regression Time: {} seconds'.format(end_time-start_time))
		# Plot
		fig = plt.figure()
		plt.plot(x[:,:-1],y,'bo', label='Data')
		plt.plot(x[:,:-1],y_fitted,'r', label='OLS Fitting')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('1D Ordinary Least Squares Demostration')
		plt.legend()
		fig.savefig('test_OLS.png', bbox_inches='tight')

	# Fit command
	if args.command=='fit':
		# Data settings
		n = args.n
		i = args.i
		x,y = generate_data(i,n)
		# Object
		start_time = time.time()
		LR = LinearRegression(x,y)
		y_fitted = LR.fit(x,y)
		end_time = time.time()
		# Fitted weights
		print('Fitted weights vector: {}'.format(LR.fitted_weigths))
		print('Elapsed Linear Regression Time: {} seconds'.format(end_time-start_time))