from setuptools import setup

setup(
    name='pyRssa',
    package_dir={"": "classes"},
    packages=['pyRssa.pyRssa'],
    version='1.0',
    description='Rssa for Python',
    author='Fleyderer',
    install_requires=['rpy2', 'pandas', 'numpy', 'matplotlib']
)
