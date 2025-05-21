from setuptools import setup, find_packages

setup(
    name="auto-sklearn2",
    version="1.0.0",
    description="Python 3.11+ compatible version of auto-sklearn",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Agnel Vishal",
    author_email="agnelvishal@gmail.com",
    url="https://github.com/agnelvishal/auto_sklearn2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
    ],
    keywords="machine learning, auto-ml, automated machine learning, python 3.11, python 3.12, python 3.13",
)
