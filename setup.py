from setuptools import find_packages, setup
NAME = 'compositional_elements'
VERSION = '0.1.0'
setup(
    name=NAME,
    packages=find_packages(include=['compositional_elements']),
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