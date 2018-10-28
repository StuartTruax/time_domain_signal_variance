# Analyzing Time-Domain Signal Variance using Numpy


This tutorial is an introduction to the use of the python numerical computing package `numpy` for computation of time-domain variance of signals, namely the Allan variance $\sigma^{2}(\tau)$. Allan variance is an important measure of statistical variance as a function of time, and is widely used to identify random signal-generating processes from sensors, clocks and other temporal data. 

The tutorial will review some essential methods of using fast fourier transforms (FFTs) and generating power-law noises in `numpy`, followed by an explanation and example of the Allan variance metric. 

For a library of read-to-use Allan variance estimators using `numpy`, please see _[AllanTools](https://pypi.org/project/AllanTools/)_.

## Getting Started

Files:

- `NumpySignalVariance.ipynb` : IPython notebook and main tutorial. 
- `AllanVariance.py`: Class with static method for calculating the non-overlapped Allan variance.
- `NoiseGenerator.pyc` :  Class with static method for generating time-domain signal vectors of power-law noises. 

### Prerequisites

Python 2.7

Jupyter 

Python Libraries: 

- `numpy`
- `matplotlib.pyplot`


## Running the code

 1. Download the repository
 2. In the repo directory, enter into the command line: `jupyter notebook`
 3. Go through the sections in the `NumpySignalVariance.ipynb` notebook

## Authors

* **Stuart Truax** - *Initial work* - (https://github.com/StuartTruax)

This work contains information soley from public and open sources. 

## License

This project is licensed under the MIT License. 


## Notebook contents (Static HTML Render)

[Contribution guidelines for this project](NumpySignalVariance.html)




