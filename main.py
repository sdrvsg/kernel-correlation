from pointcloud import PointCloud
from kernel import GaussianKernelCorrelaton
import numpy as np


def main():
    model = 'model'
    models = {
        'source': {
            'offset': [0.045, -0.045, 10.0],
            'rotate': [0, 0, 0],
            'point_color': (255, 255, 255),
            'pointcloud': None,
        },
        'transformed': {
            'offset': [0.015, -0.145, 15.0],
            'rotate': [0, 0, 0],
            'point_color': (255, 0, 0),
            'pointcloud': None,
        },
    }

    for name, data in models.items():
        cloud = PointCloud(data.get('offset'), data.get('rotate'))
        cloud.parse(f'objects/{model}.obj')
        cloud.draw(point_color=data.get('point_color'))
        # cloud.show(name)
        cloud.save(f'images/{model}_{name}.jpg')
        models[name]['pointcloud'] = cloud

    source_pc: PointCloud = models.get('source', {}).get('pointcloud', PointCloud([], []))
    transoformed_pc: PointCloud = models.get('transformed', {}).get('pointcloud', PointCloud([], []))

    kc = GaussianKernelCorrelaton(
        source_pc.points,
        transoformed_pc.points,
    )

    theta = kc.minimize()
    print(f'theta = {theta}')

    true_offset = np.array(models['source']['offset']) - np.array(models['transformed']['offset'])
    print(f'true offset = {true_offset}')

    source_pc.copy(transoformed_pc)
    source_pc.transform(theta)
    source_pc.draw(models['transformed']['point_color'])
    source_pc.show('Transformed Reverse')


if __name__ == '__main__':
    main()
