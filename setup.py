import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gdtest", 
    version="0.0.1",
    author="Fan Zhang",
    author_email="fxz89@case.edu",
    description="Gradient Descent solver package for EPRI interview",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fxz89",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)