from setuptools import setup, find_packages

setup(
    name="src",
    version="0.1.0",
    package_data={
        '':['*.yaml', '*.sh'],
    },
    packages=find_packages(),
)
