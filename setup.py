from setuptools import find_packages, setup

install_requires = [
    "datasets==2.4.0",
    "diffusers==0.3.0",
    "transformers==4.21.3",
    "h5py"
]

setup(
    name="diffusion-planner",
    version="0.0",
    package_dir={"": "diffusion_planner"},
    packages=find_packages("diffusion_planner"),
    install_requires=install_requires
)
