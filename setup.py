from setuptools import setup

setup(
    name='pyrssa',
    package_dir={"": "classes"},
    packages=['pyrssa.pyrssa'],
    version='1.0',
    description='Rssa for Python',
    author='Fleyderer',
    install_requires=['rpy2', 'pandas', 'numpy', 'matplotlib']
)
