from setuptools import setup

setup(
    name='pyrain',
    version='0.0.3',
    packages=['pyrain'],
    url='https://github.com/darshanbaral/pyrain',
    license='MIT',
    author='Darshan Baral',
    author_email='darshanbaral@gmail.com',
    description='package for stochastic rainfall generation',
    install_requires=["pandas", "numpy", "scipy"]
)
