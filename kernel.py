import math
import numpy
import scipy


class KernelCorrelation:
	def __init__(self, scene, model):
		self.counter = 0
		self.scene = scene
		self.model = model

	def kernel(self, x, center):
		pass

	def point2point_correlation(self, i, j):
		pass

	def point2cloud_correlation(self, x, cloud):
		pass

	def cloud_correlation(self, cloud):
		pass

	def cost(self, theta):
		theta = theta.reshape(self.scene.shape[1], self.model.shape[1])
		return -sum(self.point2cloud_correlation(theta.dot(i), self.scene) for i in self.model)

	def minimize(self, max_iters=100000):
		return scipy.optimize.minimize(
			self.cost,
			numpy.zeros((self.scene.shape[1] * self.model.shape[1], )),
			method='Powell',
			tol=1e-5,
			options={'maxiter': max_iters, 'maxfev': max_iters, 'disp': True}
		).x.reshape(self.scene.shape[1], self.model.shape[1])


class GaussianKernelCorrelaton(KernelCorrelation):
	def __init__(self, scene, model, sigma=1):
		super().__init__(scene, model)
		self.sigma = sigma

	def kernel(self, x, center):
		d = x.shape[0]
		diff = x - center
		return (math.pi * self.sigma ** 2) ** (-d / 2) * math.exp(-diff.dot(diff) ** 2 / (self.sigma ** 2))

	def point2point_correlation(self, i, j):
		d = i.shape[0]
		diff = i - j
		return (2 * math.pi * self.sigma ** 2) ** (-d / 2) * math.exp(-diff.dot(diff) ** 2 / (2 * self.sigma ** 2))

	def point2cloud_correlation(self, x, cloud):
		return sum(0 if (x == i).all() else self.point2point_correlation(x, i) for i in cloud)

	def cloud_correlation(self, cloud):
		return sum(self.point2cloud_correlation(i, cloud) for i in cloud)