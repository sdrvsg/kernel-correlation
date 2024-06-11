import numpy as np
import open3d as o3d
from open3d.cpu.pybind.utility import Vector3dVector


class PointCloud:
    def __init__(self, scale, offset, rotate, color):
        self.cloud = None
        self.scale = scale
        self.offset = offset
        self.rotate = rotate
        self.color = color

    @property
    def points(self):
        return np.asarray(self.cloud.points)

    def translate(self, theta):
        self.cloud.translate(theta)

    def rotate(self, theta):
        self.cloud.rotate(theta)

    def parse(self, filename):
        self.cloud = o3d.io.read_point_cloud(filename)
        colors = np.repeat(np.array([self.color]).astype(np.float_) / 255.0, np.asarray(self.cloud.points).shape[0], axis=0)
        self.cloud.colors = Vector3dVector(colors)

    def show(self):
        o3d.visualization.draw_geometries(
            [self.cloud],
            width=500,
            height=500,
        )

    @staticmethod
    def visialize(*pcs):
        o3d.visualization.draw_geometries(
            [pc.cloud for pc in pcs],
            width=500,
            height=500,
        )
