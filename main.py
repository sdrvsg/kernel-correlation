from pointcloud import PointCloud
from kernel import GaussianKernelCorrelaton


def main():
    model = 'model'
    models = {
        'source': {
            'offset': [0.005, -0.045, 15.0],
            'rotate': [0, 0, 0],
            'point_color': (255, 255, 255),
            'pointcloud': None,
        },
        'transformed': {
            'offset': [0.005, -0.045, 15.0],
            'rotate': [0, 0, 45],
            'point_color': (255, 0, 0),
            'pointcloud': None,
        },
    }

    for name, data in models.items():
        cloud = PointCloud(data.get('offset'), data.get('rotate'), point_color=data.get('point_color'))
        cloud.parse(f'objects/{model}.obj')
        cloud.draw()
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
    print(theta)
    transoformed_pc.transform(theta)

    transoformed_pc.draw()
    transoformed_pc.show('Transformed Reverse')


if __name__ == '__main__':
    main()
