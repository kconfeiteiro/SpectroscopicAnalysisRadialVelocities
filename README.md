# EP 425 Final Project Code

## How to to use this package

### Import

Importing this package is pretty simple. Follow these steps:

1. Import entire code using `pip install git+https://github.com/[kconfeiteiro]/[EP-425-Final-Project-Code]@[main]`
2. Now, you need to import the functions to read, plot, and make the calculations. For that, use `from ProjectMain.Package import AnalyzeSpectra`

#### Example

##### Code

```python
from ProjectMain import Package as spec
import os

# define root folder and indvidual files you want to plot (separetely)
file = 'HD12871704_15_23_850_20230415064551_47.fit'
root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
path = os.path.join(root, file) # creates a combined path

# initialie class and perform operations
Data = spec.AnalyzeSpectra(root) # initialize class
Data.plot_full_spectra(print=True) # plots entire combined spectra
```

##### Code Outline

1. `spec.AnalyzeSpectra(root, file)` initializes the class (where the functions are)
2. `root` is the file path that leads to your files
3. `file` is a single file that you can plot instead of the entire combined spectra
4. `Data.plot_full_spectra()` is where you plot the entire spectra, print=True tells the code to show the plot

## Project Description

### Purpose

This code is for the final project for EP 425, Observational Astronomy, for Spring 2023.

### Project Objective

Our main objective is to fit a periodic curve to shifting absorption lines (like Hydrogen and Calcium lines) of the system’s spectra to confirm the presence of exoplanets. We are looking at the star system *HD 128717* from theGAIA DR3 Archival Database. Our target system was observed with Gaia as an exoplanet candidate using photometry.

### Team

* Johnathen Saier
* Krystian Confeiteiro

## Software

### Packages used

* `SciPy`
* `Numpy`
* `MatPlotLib`
* `OS`

### Citations

1. Pauli Virtanen, Ralf Gommers, SciPy 1.0 Contributors, et. al. (2020)  SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python .  *Nature Methods* , 17(3), 261-272.
2. Harris, C.R., Millman, K.J., van der Walt, S.J. et al.  *Array programming with NumPy* . Nature 585, 357–362 (2020). DOI: [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
3. J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007 ([Publisher link](https://doi.org/10.1109/MCSE.2007.55)).

#### Code Contributers

* Krystian Confeiteiro

#### Observations

* All completed by John Saier

#### Data Reduction and Calibration

* Completed by John Saier using Demetra
