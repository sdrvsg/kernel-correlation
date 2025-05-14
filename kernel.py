import math
import scipy
import numpy as np
from numba import jit
from open3d.cpu.pybind.geometry import get_rotation_matrix_from_axis_angle
from open3d.cpu.pybind.geometry import PointCloud as PC
from open3d.cpu.pybind.utility import Vector3dVector
from pointcloud import PointCloud


class KernelCorrelation:
	def __init__(self, scene, model):
		self.scene = scene
		self.model = model

	def kernel(self, x, center):
		pass

	def point2point_correlation(self, i, j):
		return self.kernel(i, j)

	def point2cloud_correlation(self, x, cloud):
		return sum(self.point2point_correlation(x, i) for i in cloud[::100])

	def point2cloud_loo_correlation(self, x, cloud):
		pass

	def cloud_correlation(self, scene, model):
		return sum(self.point2cloud_correlation(i, scene) for i in model[::100])

	def cost(self, theta):
		offset = theta[0:3]
		angles = theta[3:6]
		r = get_rotation_matrix_from_axis_angle(angles)
		return -self.cloud_correlation(self.scene, self.model @ r.T + offset)

	def minimize(self, max_iters=10 ** 6):
		x0 = np.zeros(6)
		return scipy.optimize.minimize(
			self.cost,
			x0,
			method='L-BFGS-B',
			tol=1e-6,
			options={
				'maxiter': max_iters,
				'disp': True
			},
		).x


class GaussianKernelCorrelation(KernelCorrelation):
	def __init__(self, scene, model, sigma=1, block_size=2000):
		super().__init__(scene, model)
		self.sigma = sigma
		self.block_size = block_size

	def kernel(self, x, center):
		# (2 * math.pi * self.sigma ** 2) ** (-x.shape[0] / 2) *
		diff = x - center
		return math.exp(-diff.dot(diff) / (2 * self.sigma ** 2))

	def point2cloud_loo_correlation(self, x, cloud):
		return sum(0 if (x == i).all() else self.point2point_correlation(x, i) for i in cloud)

	def cost(self, theta):
		offset = theta[0:3]
		angles = theta[3:6]
		r = get_rotation_matrix_from_axis_angle(angles)
		print(offset, angles)

		s = PointCloud(color=(255, 0, 0))
		s.cloud = PC()
		s.cloud.points = Vector3dVector(self.scene[::10])

		m = PointCloud(color=(0, 0, 255))
		m.cloud = PC()
		m.cloud.points = Vector3dVector(self.model[::10])
		m.angle = angles
		m.offset = offset
		# PointCloud.visualize(s, m)
		return -self.cloud2cloud_correlation(PointCloud.rotate_by_model_center(self.model[::10], r), self.scene[::10], self.sigma, self.block_size)

	@staticmethod
	@jit(nopython=True)
	def cloud2cloud_correlation(first, second, sigma, block_size):
		divisor = 2 * sigma ** 2
		summary = 0.0
		norms = np.zeros((block_size, block_size))

		square_i = (first.shape[0] - 1) / block_size + 1
		square_j = (second.shape[0] - 1) / block_size + 1

		# Тут пробег по всем полным блокам размера block_size на block_size
		for s_i in range(square_i - 1):
			for s_j in range(square_j - 1):
				# и внутри стараемся считать всё максимально функциями numpy
				# заполняем квадраты норм разностей поэлементно
				for i in range(block_size):
					for j in range(block_size):
						diff = second[j + s_j*block_size] - first[i + s_i * block_size]
						norms[i, j] = diff.dot(diff)
				# а экспоненты и суммы считаем сразу для всего
				exp_res = np.exp(- norms / divisor)
				summary += np.sum(exp_res)

		# Вообще тут надо ещё последний неполный блок пробежать, но мне стало лень
		return summary
