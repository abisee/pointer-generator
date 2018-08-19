from setuptools import setup, find_packages

__version__ = '1.0'

setup(
    name='sgcharts-pointer-generator',
    version=__version__,
    python_requires='>=3.6.0',
    install_requires=[
        'tensorflow==1.10.0',
        'pyrouge==0.1.3'
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    include_package_data=True,
    description='News Summarizer'
)
