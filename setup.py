from setuptools import find_packages, setup
setup(
    name='compositional_elements',
    packages=find_packages(include=['compositional_elements']),
    version='0.1.0',
    description='Library for generating and comparing compositional elements from art historic images.',
    author='Tilman Marquart',
    license='MIT',
    python_requires='>=3.8',
    install_requires=['opencv-python','numpy','typing'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)