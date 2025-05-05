import math
import numpy
import numpy as np
import scipy
from open3d.cpu.pybind.geometry import get_rotation_matrix_from_xyz


class KernelCorrelation:
	def __init__(self, scene, model):
		self.scene = scene
		self.model = model

	def kernel(self, x, center):
		pass

	def point2point_correlation(self, i, j):
		pass

	def point2cloud_correlation(self, x, cloud):
		pass

	def point2cloud_loo_correlation(self, x, cloud):
		pass

	def cloud_correlation(self, cloud):
		pass

	def cost(self, theta):
		return -sum(self.point2cloud_correlation(theta[0:3] + i, self.scene) for i in np.asarray(self.model)[:100])

	def minimize(self, max_iters=100000):
		return scipy.optimize.minimize(
			self.cost,
			numpy.zeros((6, )),
			method='Powell',
			tol=1e-5,
			options={'maxiter': max_iters, 'maxfev': max_iters, 'disp': True}
		).x


class GaussianKernelCorrelation(KernelCorrelation):
	def __init__(self, scene, model, sigma=1):
		super().__init__(scene, model)
		self.sigma = sigma

	def kernel(self, x, center):
		d = x.shape[0]
		diff = x - center
		return (math.pi * self.sigma ** 2) ** (-d / 2) * math.exp(-diff.dot(diff) ** 2 / (self.sigma ** 2))

	def point2point_correlation(self, i, j):
		# d = i.shape[0]
		diff = i - j
		# return (2 * math.pi * self.sigma ** 2) ** (-d / 2) * math.exp(-diff.dot(diff) ** 2 / (2 * self.sigma ** 2))
		return math.exp(-diff.dot(diff) ** 2 / (2 * self.sigma ** 2))

	def point2cloud_correlation(self, x, cloud):
		return sum(self.point2point_correlation(x, i) for i in np.asarray(cloud)[:100])

	def point2cloud_loo_correlation(self, x, cloud):
		return sum(0 if (x == i).all() else self.point2point_correlation(x, i) for i in np.asarray(cloud))

	def cloud_correlation(self, cloud):
		return sum(self.point2cloud_correlation(i, cloud) for i in np.asarray(cloud))
