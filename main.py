from contextlib import contextmanager
from time import perf_counter
import numpy as np
from pointcloud import PointCloud
from kernel import GaussianKernelCorrelation


@contextmanager
def counter():
    start = perf_counter()
    yield
    end = perf_counter()
    print(f'Выполнено за {end - start:.6f} сек')


def main():
    model = 'objects/bunny.ply'
    model_scaling = (55000, -55000)
    percent = 1
    sigma = 0.1

    source_offset = (0.045, -0.045, 10.0)
    source_angle = (0.0, 0.0, 0.0)
    source_color = (0, 255, 0)

    transformed_offset = (0.045, -0.045, 10.5)
    transformed_angle = (np.radians(35.0), np.radians(-100.0), np.radians(35.0))
    transformed_color = (255, 0, 0)

    source_cloud = PointCloud(model, model_scaling, source_offset, source_angle, source_color)
    source_cloud.remove_random(percent)

    transformed_cloud = PointCloud(model, model_scaling, transformed_offset, transformed_angle, transformed_color)
    transformed_cloud.remove_random(percent)

    PointCloud.visualize(source_cloud, transformed_cloud)

    kc = GaussianKernelCorrelation(
        source_cloud.points,
        transformed_cloud.points,
        sigma=sigma
    )

    with counter():
        theta = kc.minimize()

    offset_slice = (0, 3)
    rotate_slice = (3, 6)
    # rotate_slice = (0, 3)

    true_offset = np.array(source_offset) - np.array(transformed_offset)
    offset_diff = true_offset - theta[offset_slice[0]:offset_slice[1]]
    true_offset_norm = np.linalg.norm(true_offset)
    offset_percentage = ((np.linalg.norm(offset_diff) / true_offset_norm) if true_offset_norm else 0) * 100

    true_angle = np.array(source_angle) - np.array(transformed_angle)
    angle_diff = true_angle - theta[rotate_slice[0]:rotate_slice[1]]
    true_angle_norm = np.linalg.norm(true_angle)
    angle_percentage = ((np.linalg.norm(angle_diff) / true_angle_norm) if true_angle_norm else 0) * 100

    print(f'Полученное смещение: {theta[offset_slice[0]:offset_slice[1]]}')
    print(f'Настоящее смещение: {true_offset}')
    print(f'Разница: {offset_diff} ({offset_percentage:.3f}%)')
    print()
    print(f'Полученное вращение: {theta[rotate_slice[0]:rotate_slice[1]]}')
    print(f'Настоящее вращение: {true_angle}')
    print(f'Разница: {angle_diff} ({angle_percentage:.3f}%)')

    transformed_cloud.offset = theta[offset_slice[0]:offset_slice[1]]
    transformed_cloud.angle = theta[rotate_slice[0]:rotate_slice[1]]
    PointCloud.visualize(source_cloud, transformed_cloud)


if __name__ == '__main__':
    main()
