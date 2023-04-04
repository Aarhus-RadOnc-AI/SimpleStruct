from setuptools import setup, find_namespace_packages

setup(
    name='simplestruct',
    version='1',
    description='Toolbox for structures',
    author='Mathis Rasmussen',
    author_email='mathis.rasmussen@rm.dk',
    url='https://github.com/mathiser/SimpleStruct',
    python_requires=">=3.10",
    packages=find_namespace_packages(include=["simplestruct", "simplestruct.*"]),
    install_requires=[
        "pydantic>=1.10.7",
        "simpleitk>=2.2.1",
        "numpy>=1.23.5",
        "pydicom>=2.3.1",
        "numba>=0.56.4",
        "scikit-image>=0.2.0"
    ]
)
