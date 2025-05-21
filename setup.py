from setuptools import setup, find_packages

setup(
    name="auto-sklearn2",
    version="0.1.0",
    description="Python 3.13 compatible version of auto-sklearn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Agnel Vishal",
    author_email="agnelvishal@gmail.com",
    url="https://github.com/agnelvishal/auto_sklearn2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.13",
    install_requires=[
        "numpy>=2.0.0",
        "scipy>=1.15.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
    ],
    keywords="machine learning, auto-ml, automated machine learning, python 3.13",
)
