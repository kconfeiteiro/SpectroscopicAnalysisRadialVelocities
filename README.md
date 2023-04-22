# Observational Astronomy (EP425) Final Project

## Project Details

### Purpose

This code is for the final project for EP 425, Observational Astronomy, for Spring 2023 at *Embry-Riddle Aeronautical University*.

### Objective

Our main objective is to fit a periodic curve to shifting absorption lines (like Hydrogen and Calcium lines) of the system's spectra then calculate radial velocities to confirm the presence of exoplanets. We are looking at the star system *HD 128717* from the GAIA DR3 Archival Database. The radial velocity will be calculated after the change in wavelength, from which we can confirm the existence of the exoplanet surrounding HD 128717.

### Team

* John Saier
  * Observations, data reduction in Demetra
* Krystian Confeiteiro
  * Data Analysis in Python

# How to to use this package

## Import

Importing this package is pretty simple. Follow these steps:

1. Import entire code using `pip install git+https://github.com/[kconfeiteiro]/[EP-425-Final-Project-Code]@[main]`
2. Now, you need to import the functions to read, plot, and make the calculations. For that, use `from ProjectMain.Package import AnalyzeSpectra`
   * For other functions, you will follow the same for importing and calling them.

## Issues

If you are having issue pip installing manually, please run the `setup.bat` file by opening up your command prompt and typing `setup.bat`

## Example

Please see main.py ([link](https://github.com/kconfeiteiro/EP-425-Final-Project-Code/blob/main/main.py)) for the full example.

### Code

```python
from ProjectMain.Package import AnalyzeSpectra
import os

# define root folder and indvidual files you want to plot (separetely)
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

* Demetra (calibration and reduction)
* Python (analysis)

## Packages used

* `SciPy`
* `Numpy`
* `MatPlotLib`
* `OS`

# Citations

1. Pauli Virtanen, Ralf Gommers, SciPy 1.0 Contributors, et. al. (2020)  SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python .  *Nature Methods* , 17(3), 261-272.
2. Harris, C.R., Millman, K.J., van der Walt, S.J. et al.  *Array programming with NumPy* . Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
3. J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007 ([Publisher link](https://doi.org/10.1109/MCSE.2007.55)).

# Issues

If you have any issues with the code, please email me at [confeitk@my.erau](mailto:confeitk@my.erau.edu).
