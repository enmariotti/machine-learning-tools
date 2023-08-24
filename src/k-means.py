# Instituto Balseiro
# Redes Neuronales y Aprendizaje Profundo para Visión Artificial - 2019
# Alumno:  Enrique Nicanor Mariotti
# Carrera: Maestria en Cs de la Ingenieria

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

class KMeans():

	def __init__(self, data, k, tol=0.1, max_iter=400):
		self.i = data.shape[0] # Number of n-dimensional points
		self.n = data.shape[1] # n-dimension
		self.k = k # Number of clusters
		self.tol = tol # Tolerance to stop
		self.max_iter = max_iter # Iterations to stop
		self.centroids = None

	def fit(self,data):
		start_time = time.time()
		# Initialize centroids
		self.__init_centroids__()

		# Classification Kernel
		for current_iter in range(self.max_iter):

			# Classification dictionary
			self.__init_classifications__()

			#Clustering
			self.__clustering__(data)

			# Old centroids
			old_centroids = dict(self.centroids)

			# Update centroids
			self.__update_centroids__()

			# Check for convergence
			stop = True
			print('\nCurrent iteration: {}'.format(current_iter))
			for cluster_index in self.centroids:
				original_centroid = old_centroids[cluster_index]
				current_centroid = self.centroids[cluster_index]
				error = np.sum(np.abs((current_centroid - original_centroid)/ original_centroid)*100.0)
				print('Cluster {} centroid movement: {} %'.format(cluster_index, error))
				if error > self.tol:
					stop = False
			
			if stop:
				end_time = time.time()
				print('\nElapsed K Means Time: {} seconds'.format(end_time-start_time))
				#Re-shape clusters
				for i in range(self.k):	
					self.classifications[i] = np.vstack(self.classifications[i])
				break
		return self.centroids, self.classifications

	def predict(self, point):
		assert self.centroids is not None, 'Fit method needs to be called first'
		euclidean_distances = [np.linalg.norm(point - self.centroids[centroid], ord=2) for centroid in self.centroids]
		cluster_index = np.argmin(euclidean_distances)
		return cluster_index

	def __init_centroids__(self):
		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i] #initialize within the data

	def __init_classifications__(self):
		self.classifications = {}
		for i in range(self.k):
			self.classifications[i] = []

	def __clustering__(self,data):
		for point in data:
			euclidean_distances = [np.linalg.norm(point - self.centroids[centroid], ord=2) for centroid in self.centroids]
			cluster_index = np.argmin(euclidean_distances)
			self.classifications[cluster_index].append(point)

	def __update_centroids__(self):
		for cluster_index in self.classifications:
			self.centroids[cluster_index] = np.average(self.classifications[cluster_index], axis=0)

def generate_data(i,n,p):
	# For reproducibility
	np.random.seed(1988)
	# Gaussian data
	mu = np.random.uniform(0,10,p)
	sigma = np.random.uniform(0,10,p)
	data_list = []
	for mu, sigma in zip(mu, sigma):		
		data_list.append(np.random.normal(mu, sigma, (i,n)))
	data = np.vstack(data_list)
	return data

if __name__ == "__main__":

	# Parsers & sub-parsers
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command', help='test (for 2D demostration) or fit (for N-D fitting)')

	# # Fit
	fit = subparsers.add_parser('fit', help='N-D fitting')
	fit.add_argument('-i', type=int, metavar='points per distribution', help='points per distribution', default=100)
	fit.add_argument('-n', type=int, metavar='n-dimension', help='n-dimension', default=3)
	fit.add_argument('-p', type=int, metavar='p-gaussians', help='p-gaussians', default=4)
	fit.add_argument('-k', type=int, metavar='k-means', help='k-means', default=4)

	# Test
	test = subparsers.add_parser('test', help='2D demostration')

	# Dicttionary of parsers
	args = parser.parse_args()

	# Test command
	if args.command=='test':
		# Data settings
		data = generate_data(100,2,3) # 0ne hundred 2-dimensional points from each of 3 gaussian distributions
		k = 3 #k-clusters
		# K Means
		KM = KMeans(data, k)
		means, clusters = KM.fit(data)
		# Plot
		fig = plt.figure()
		for cluster_index in range(k):
			plt.scatter(clusters[cluster_index][:,0],clusters[cluster_index][:,1], label=cluster_index)
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('2D K-Means Clustering Example')
		plt.legend()
		fig.savefig('test_K_means.png', bbox_inches='tight')

	# Fit command
	if args.command=='fit':
		# Data settings
		i = args.i
		n = args.n
		p = args.p
		k = args.k
		# Data settings
		data = generate_data(i,n,p) # "i" n-dimensional points from each of p gaussian distributions
		# K Means
		KM = KMeans(data, k)
		means, clusters = KM.fit(data)
		# Prediction
		example = np.random.uniform(0,10,n)
		prediction = KM.predict(example)
		print('Predicted cluster for vector {}: {}'.format(example,prediction)) 