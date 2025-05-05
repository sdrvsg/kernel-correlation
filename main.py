from pointcloud import PointCloud
from kernel import GaussianKernelCorrelation
import numpy as np


def main():
    model = 'bunny'
    ext = 'ply'

    models = {
        'source': {
            'scale': [55000, -55000],
            'offset': [0.045, -0.045, 10.0],
            'rotate': [0, 0, 0],
            'color': [0, 255, 0],
            'pointcloud': None,
        },
        'translated': {
            'scale': [55000, -55000],
            'offset': [0.015, -0.145, 15.0],
            'rotate': [0, 0, 0],
            'color': [255, 0, 0],
            'pointcloud': None,
        },
        'rotated': {
            'scale': [55000, -55000],
            'offset': [0.045, -0.045, 10.0],
            'rotate': [np.pi, 0.0, 0.0],
            'color': [255, 0, 0],
            'pointcloud': None,
        },
    }

    percent = 0.5
    for name, data in models.items():
        cloud = PointCloud(data.get('scale'), data.get('offset'), data.get('rotate'), data.get('color'))
        cloud.parse(f'objects/{model}.{ext}')
        cloud.remove_random(percent)
        models[name]['pointcloud'] = cloud

    source_pc: PointCloud = models.get('source').get('pointcloud', PointCloud())
    translated_pc: PointCloud = models.get('translated').get('pointcloud', PointCloud())

    kc = GaussianKernelCorrelation(
        source_pc.points,
        translated_pc.points,
    )

    theta = kc.minimize()
    true_offset = np.array(models['source']['offset']) - np.array(models['translated']['offset'])
    true_rotate = np.array(models['source']['rotate']) - np.array(models['translated']['rotate'])
    offset_diff = true_offset - theta[0:3]
    offset_percentage = np.linalg.norm(offset_diff)

    print(f'offset = {theta[0:3]}')
    print(f'true offset = {true_offset}')
    print(f'offset diff = {offset_diff}, percentage = {offset_percentage}')
    print(f'rotate = {theta[3:6]}')
    print(f'true rotate = {true_rotate}')

    PointCloud.visualize(source_pc, translated_pc)
    translated_pc.translate(theta[0:3])
    translated_pc.rotate(theta[3:6])
    PointCloud.visualize(source_pc, translated_pc)


if __name__ == '__main__':
    main()
