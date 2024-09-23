from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name="mike21modeltools",
    version="1.0",
    description="Mike21 model tools",
    author="Tianju Du",
    packages=find_packages(),
    install_requires=requirements,
)
