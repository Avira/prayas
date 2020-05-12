from setuptools import setup
from prayas import __version__

with open('requirements-minimal.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='prayas',
    version=__version__,
    description='Bayesian A/B Testing',
    license='MIT',
    packages=['prayas'],
    author='Data Analytics & Insights @ Avira Operations GmbH & Co. KG',
    keywords=['prayas'],
    url='https://avira.github.io/prayas',
    install_requires=requirements,
    python_requires='>=3.7'
)