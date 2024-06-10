from pointcloud import PointCloud
from kernel import GaussianKernelCorrelaton
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
        'transformed': {
            'scale': [55000, -55000],
            'offset': [0.015, -0.145, 15.0],
            'rotate': [0, 0, 0],
            'color': [255, 0, 0],
            'pointcloud': None,
        },
    }

    for name, data in models.items():
        cloud = PointCloud(data.get('scale'), data.get('offset'), data.get('rotate'), data.get('color'))
        cloud.parse(f'objects/{model}.{ext}')
        # cloud.show()
        models[name]['pointcloud'] = cloud

    source_pc = models.get('source').get('pointcloud')
    transoformed_pc = models.get('transformed').get('pointcloud')

    kc = GaussianKernelCorrelaton(
        source_pc.points,
        transoformed_pc.points,
    )

    theta = kc.minimize()
    print(f'theta = {theta}')

    true_offset = np.array(models['source']['offset']) - np.array(models['transformed']['offset'])
    print(f'true offset = {true_offset}')

    transoformed_pc.translate(theta)
    PointCloud.visialize(source_pc, transoformed_pc)


if __name__ == '__main__':
    main()
