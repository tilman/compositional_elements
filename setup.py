from setuptools import find_packages, setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}
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
    install_requires=['opencv-python','numpy','typing','shapely','yaml'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    cmdclass=cmdclass,
    command_options={
        'build_sphinx': {
            'project': ('setup.py', NAME),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', VERSION),
            'source_dir': ('setup.py', 'doc')
        }
    },
)