import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import get_rotation_matrix_from_xyz
from open3d.cpu.pybind.utility import Vector3dVector


class PointCloud:
    def __init__(self, scaling=(1, 1), offset=(0.0, 0.0, 0.0), angle=(0.0, 0.0, 0.0), color=(0, 0, 0)):
        self.cloud = None
        self.scaling = np.array(scaling)
        self.offset = np.array(offset)
        self.angle = np.array(angle)
        self.color = color

    @property
    def points(self):
        return np.asarray(self.cloud.points)

    def translate(self, offset):
        self.cloud.translate(offset)

    def rotate(self, angle):
        matrix = get_rotation_matrix_from_xyz(angle)
        self.cloud.rotate(matrix)

    def scale(self, scaling):
        self.cloud.scale(scaling)

    def parse(self, filename):
        self.cloud = o3d.io.read_point_cloud(filename)
        colors = np.repeat(np.array([self.color]).astype(np.float64) / 255.0, np.asarray(self.cloud.points).shape[0], axis=0)
        self.cloud.colors = Vector3dVector(colors)

        self.translate(self.offset)
        self.rotate(self.angle)

    def remove_random(self, p=0.1):
        self.cloud = self.cloud.random_down_sample(p)

    def remove_slice(self, p, left):
        points = np.asarray(self.cloud.points)
        x_coords = points[:, 0]  # Все X-координаты
        offset_mul = p if left else 1

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        threshold = x_min + offset_mul * (x_max - x_min)  # Удаляем первые 30% по ширине
        filtered_points = points[x_coords > threshold] if left else points[x_coords < threshold]
        print(filtered_points.shape[0])

        # Создаем новое облако точек
        self.cloud = o3d.geometry.PointCloud()
        self.cloud.points = o3d.utility.Vector3dVector(filtered_points)

        colors = np.repeat(np.array([self.color]).astype(np.float64) / 255.0, np.asarray(self.cloud.points).shape[0], axis=0)
        self.cloud.colors = Vector3dVector(colors)

    def show(self):
        o3d.visualization.draw_geometries(
            [self.cloud],
            width=500,
            height=500,
        )

    @staticmethod
    def visualize(*pcs):
        o3d.visualization.draw_geometries(
            [pc.cloud for pc in pcs],
            width=500,
            height=500,
        )
