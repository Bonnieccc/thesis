# Wave filtering from paper "Learning Linear Dynamical Systems via Spectral Filtering", 2017
# Author: John Hallman

import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# compute and return top k eigenpairs of a TxT Hankel matrix
def eigen_pairs(T, k):
	v = np.fromfunction(lambda i: 1.0 / ((i+2)**3 - (i+2)), (2 * T - 1,))
	Z = 2 * la.hankel(v[:T], v[T-1:])
	eigen_values, eigen_vectors = np.linalg.eigh(Z)
	return eigen_values[-k:], eigen_vectors[:,-k:]


# Wave filtering with squared loss
def wave_filter(X, Y, k, k_values, k_vectors, eta=0.00002, verbose=False):

	R_Theta = 5
	R_M = 2 * R_Theta * R_Theta * k**0.5 

	# the total number of time steps.
	T = X.shape[1]
	# the dimension of the input vectors.
	n = X.shape[0]
	# the dimension of the output vectors.
	m = Y.shape[0]
	if (k > T):
		raise Exception("Model parameter k larger than timespan T")

	# initialize M_1
	k_prime = n * k + 2 * n + m
	M = 2 * np.random.rand(m, k_prime) - 1

	# iterate over data points in Y
	losses = []
	y_predictions = np.zeros(Y.shape)
	for t in range(T):
		if (t == 0): # t = 0 results in an excessively complicated corner case otherwise
			X_sim = np.append(np.zeros(n * k + n), np.append(X[:,0], np.zeros(m)))
		else:
			eigen_diag = np.diag(k_values**0.25)
			X_sim_pre = X[:,0:t-1].dot(np.flipud(k_vectors[0:t-1,:])).dot(eigen_diag)
			x_y_cols = np.append(np.append(X[:,t-1], X[:,t]), Y[:,t-1])
			X_sim = np.append(X_sim_pre.T.flatten(), x_y_cols)
		
		y_hat = M.dot(X_sim)
		y_delta = Y[:,t] - y_hat
		M = M - 2 * eta * np.outer(y_delta, X_sim)
		if (np.linalg.norm(M) >= R_M):
			M = M * (R_M / np.linalg.norm(M))
			y_hat = M.dot(X_sim)
			y_delta = Y[:,t] - y_hat

		y_predictions[:,t] = y_hat
		losses.append(y_delta.dot(y_delta))

	if verbose:
		plt.subplot(221)
		# plt.plot(X[0,:], label="x")
		plt.title("Test Wave Filtering with %s, $\eta$=%f" % ("cartpole-v1", eta))
		y_index = 0
		plt.plot(Y[y_index,:], label="y[%d]" % y_index)
		plt.plot(y_predictions[y_index,:], label="yhat[%d]" % y_index)
		plt.legend()
		

		plt.subplot(222)
		y_index = 1
		plt.plot(Y[y_index,:], label="y[%d]" % y_index)
		plt.plot(y_predictions[y_index,:], label="yhat[%d]" % y_index)
		plt.legend()

		plt.subplot(223)
		y_index = 2
		plt.plot(Y[y_index,:], label="y[%d]" % y_index)
		plt.plot(y_predictions[y_index,:], label="yhat[%d]" % y_index)
		plt.legend()

		plt.subplot(224)
		y_index = 3
		plt.plot(Y[y_index,:], label="y[%d]" % y_index)
		plt.plot(y_predictions[y_index,:], label="yhat[%d]" % y_index)
		plt.legend()

		plt.tight_layout()
		plt.show()

	return losses

