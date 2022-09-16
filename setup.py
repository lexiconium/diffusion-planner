from setuptools import find_packages, setup

install_requires = [
    # "accelerate==0.12.0",
    "diffusers==0.3.0",
    "transformers==4.21.3",
    "datasets==2.4.0",
    "h5py",
    "scikit-learn"
]

setup(
    name="diffusion-planner",
    version="0.0",
    packages=find_packages(),
    install_requires=install_requires
)
