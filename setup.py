from setuptools import setup

setup(
    name='pyrssa',
    packages=['pyrssa', 'pyrssa.classes'],
    include_package_data=True,
    package_data={
        'data': ['*']
    },
    version='1.0.5',
    description='Rssa for Python',
    author='Fleyderer',
    author_email='fleyderer@gmail.com',
    url="https://github.com/Fleyderer/pyrssa",
    license="Apache 2.0",
    install_requires=['rpy2<=3.4.5', 'pandas', 'numpy', 'matplotlib']
)
