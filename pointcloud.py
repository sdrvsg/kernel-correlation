import math
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

    def draw(self, point_color=(255, 255, 255)):
        scale = [55000, -55000]
        center = [500, 600]

        for point in self.points:
            x, y, z = point[0], point[1], point[2]

            # z = self.offset[2] + z
            # x = scale[0] * (x + self.offset[0]) / z + center[0]
            # y = scale[1] * (y + self.offset[1]) / z + center[1]

            x = scale[0] * x / z + center[0]
            y = scale[1] * y / z + center[1]

            i = max(int(min(x, self.width - 1)), 0)
            j = max(int(min(y, self.height - 1)), 0)
            self.image.putpixel((i, j), point_color)

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
        return

        # Вертим сразу, оффсет и скейл потом при отображении
        alpha, beta, gamma = map(math.radians, self.rotate)
        rotate_x = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), np.sin(alpha)],
            [0, -np.sin(alpha), np.cos(alpha)],
        ])

        rotate_y = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])

        rotate_z = np.array([
            [np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        with open(filename, 'r') as file:
            counter = 0
            for line in file:
                counter += 1
                if counter == 300:
                    break

                if line:
                    if line[:2] == 'v ':
                        v, x, y, z = line.split()
                        r = np.dot(np.dot(rotate_x, rotate_y), rotate_z)
                        self.points = np.append(self.points, [list(np.dot(r, np.array([float(x), float(y), float(z)])) + np.array(self.offset))], axis=0)

        self.points = np.delete(self.points, 0, 0)

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
