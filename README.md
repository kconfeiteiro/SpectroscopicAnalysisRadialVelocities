# Determining Radial Velocities With Spectroscopy

## Project Details & Objectives

### Details

This code is for the final project for Observational Astronomy (EP-425) for the Spring 2023 semester, at *Embry-Riddle Aeronautical University*. Please note that the code for this project is _unfinished_, and I plan to finish it as soon as I can.

### Objectives

Our main objective is to fit a periodic curve to shifting absorption lines (like Hydrogen and Calcium lines) of the system's spectra and then calculate radial velocities to confirm the presence of exoplanets. We are looking at the star system *HD 128717* from the [GAIA DR3 Archival Database](https://www.cosmos.esa.int/web/gaia/dr3). The radial velocity will be calculated after the change in wavelength, from which we can confirm the existence of the exoplanet surrounding HD 128717.

### Team

* [John Saier](https://www.linkedin.com/in/jonathan-saier/)
  * Observations and data reduction using Demetra
* [Krystian Confeiteiro](https://www.linkedin.com/in/kconfeiteiro)
  * Data analysis in Python

# How To Use The Code

## Imports

Importing this package is pretty simple. Follow these steps:

1. Import the entire code using `git clone https://github.com/kconfeiteiro/SpectroscopicAnalysisRadialVelocities`
2. To `pip install`, use:
    - Push
4. Now, you need to import the functions to read, plot, and make the calculations. For that, use `from ProjectMain.Package import AnalyzeSpectra`
   * For other functions, you will follow the same for importing and calling them.

## Issues

If you are having issues with `pip` installing manually, please run the `setup.bat` file by opening up your command prompt and typing `setup.bat` (NOTE: You will need to download this manually and run it in the folder where your main code is located)

## Example

Please see [`main.py`](https://github.com/kconfeiteiro/EP-425-Final-Project-Code/blob/main/main.py) for the full example.

### Code

```py
from ProjectMain.Package import AnalyzeSpectra
import os

# define the root folder and individual files you want to plot (separately)
file = 'HD12871704_15_23_850_20230415064551_47.fit'
root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
path = os.path.join(root, file) # creates a combined path

# initialie class and perform operations
Data = AnalyzeSpectra(root) # initialize class
Data.plot_full_spectra(print=True) # plots entire combined spectra
```

### Outline

1. `spec.AnalyzeSpectra(root, file)` initializes the `AnalyzeSpectra` class
2. `Data.plot_full_spectra()` is where you plot the entire spectra, `print=True` tells the code to show the plot

# Tools

## Software

* [Demetra](https://www.shelyak.com/software/demetra/?lang=en)
* [Python](https://docs.python.org/3/library/) 

## Packages used

* `scipy`
* `numpy`
* `matplotlib`
* `os`

# Citations

1. Pauli Virtanen, Ralf Gommers, SciPy 1.0 Contributors, et. al. (2020)  SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python .  *Nature Methods* , 17(3), 261-272.
2. Harris, C.R., Millman, K.J., van der Walt, S.J. et al.  *Array programming with NumPy* . Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
3. J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007 ([Publisher link](https://doi.org/10.1109/MCSE.2007.55)).
4. Astropy Collaboration, Adrian M. Price-Whelan, Pey Lian Lim, and etl. al. Earl, Astropy Project Contributors. The Astropy Project: Sustaining and Growing a Community-oriented Open-source Project and the
Latest Major Release (v5.0) of the Core Package. apj, 935(2):167, August 2022

## Test Data

Thank Antonio Cascio and Evan Bryson for sharing their data to allow me to test the code. 

# Issues

If you have any issues with the code, please email me at [confeitk@my.erau.edu](mailto:confeitk@my.erau.edu).
