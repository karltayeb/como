from setuptools import setup, find_packages

setup(
    name='como',
    version='0.1.0',
    license='MIT',
    author="Karl Tayeb",
    url="https://github.com/karltayeb/como",
    keywords="covariate moderated empirical bayes normal means"
    packages=find_packages(include=['como', 'como.*'])
    install_requires=['jax', 'numpy', 'scipy']
)