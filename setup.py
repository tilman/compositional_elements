from setuptools import find_packages, setup
NAME = 'compoelem'
VERSION = '0.0.3'
setup(
    name=NAME,
    packages=['compoelem', 'compoelem.generate', 'compoelem.visualize', 'compoelem.detect', ],#'compoelem.compare'],
    include_package_data=True,
    version=VERSION,
    description='Library for generating and comparing compositional elements from art historic images.',
    author='Tilman Marquart',
    license='MIT',
    python_requires='>=3.8',
    install_requires=['opencv-python','numpy','typing','shapely','pyyaml','torch','torchvision','yacs','scikit-image'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)