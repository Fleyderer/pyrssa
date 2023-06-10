from setuptools import setup
from pathlib import Path

setup(
    name='pyrssa',
    packages=['pyrssa', 'pyrssa.classes'],
    include_package_data=True,
    package_data={
        'data': ['*']
    },
    version='1.0.10',
    description='Rssa for Python',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    author='Fleyderer',
    author_email='fleyderer@gmail.com',
    url="https://github.com/Fleyderer/pyrssa",
    license="Apache 2.0",
    install_requires=['rpy2', 'pandas', 'numpy', 'matplotlib']
)
