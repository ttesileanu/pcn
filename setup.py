from distutils.core import setup

setup(
    name="pcn",
    version="0.1.0",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    url="https://github.com/ttesileanu/pcn",
    packages=["pcn"],
    install_requires=[
        "numpy",
        "scipy",
        "setuptools",
        "torch",
        "torchvision",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pydove",
        "ipykernel",
    ],
)
