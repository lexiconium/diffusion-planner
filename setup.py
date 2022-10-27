from setuptools import find_packages, setup

install_requires = [
    "accelerate==0.13.2",
    "datasets",
    "diffusers==0.5.1",
    "transformers==4.23.1",
    "h5py"
]

setup(
    name="diffusion-planner",
    version="0.0",
    packages=find_packages(),
    install_requires=install_requires
)
