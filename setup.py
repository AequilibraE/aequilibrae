import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aequilibrae",
    version="0.4.2",
    author="Pedro Camargo",
    author_email="pedro@xl-optim.com",
    description="A package for transportation modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AequilibraE/pyquilibrae",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)