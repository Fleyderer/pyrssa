from setuptools import setup

setup(
    name='pyrssa',
    package_dir={"": "classes"},
    packages=['pyrssa.pyrssa'],
    version='1.0',
    description='Rssa for Python',
    author='Fleyderer',
    author_email='fleyderer@gmail.com',
    url="https://github.com/Fleyderer/pyrssa",
    license="Apache 2.0",
    install_requires=['rpy2', 'pandas', 'numpy', 'matplotlib']
)
