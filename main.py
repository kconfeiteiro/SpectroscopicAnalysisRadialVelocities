from astropy.constants.si import c, G, M_sun, R_sun
from iteration_utilities import deepflatten
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import pandas as pd
import numpy as np
import astropy, os

'''
Links
-----
1. https://astronomy.stackexchange.com/questions/43552/working-with-stellar-spectra-in-fits-format-in-python

'''

# root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
# hdul = fits.open(root + 'HD12871704_15_23_850_20230415064551_47.fit')
# data = hdul[0].data
# h1 = hdul[0].header
# obj_name = h1.get('OBJECT', 'Unknown')

# print(data)
# flux = data
# w = WCS(h1, naxis=1, relax=False, fix=False)
# lam = w.wcs_pix2world(np.arange(len(flux)), 0)[0]

# plt.plot(lam, flux)
# plt.xlabel('Wavelength (Angstrom)')
# plt.ylabel('Normalized flux')

class AnalyzeSpectra:

    def __init__(self, path):
        self.path = path

        self.directory = sorted(os.listdir(self.path))

        if '.fit' in self.path: 
            self.filename = self.path.split('/')[-1].replace('.fit','')
            self.target = self.filename.split('_')[0]
            self.title = self.target + ', Number: ' + self.filename.split('_')[-1]
        else:
            self.target = self.directory[0].split('/')[-1].replace('.fit','').split('_')[0]
            self.title = 'Partial Spectra of ' + self.target
    
    def read_single_file(self, file=None):
        if file is None:
            file = self.path

        print(file)
        sp = fits.open(file)
        header = sp[0].header

        wcs = WCS(header)
        index = np.arange(header['NAXIS1'])

        wavelength = wcs.wcs_pix2world(index[:,np.newaxis], 0)
        self.wavelength = wavelength.flatten()
        self.flux = sp[0].data

        return self.wavelength, self.flux
    
    def flatten_list(self, x):
        return list(deepflatten(x))
    
    def create_spectra(self, save=False):
        self.x, self.y = [0]*len(self.directory), [0]*len(self.directory)
        for i, file in enumerate(self.directory):
            file = os.path.join(self.path, file)
            x1, y1 = self.read_single_file(file)
            self.x[i] = x1
            self.y[i] = y1

        if save:
            return self.flatten_list(self.x), self.flatten_list(self.y)
    
    def plot_spectra(self, X=None, y=None, title=None, x_title='Wavelength (Angstroms)', y_title=' Normalized Flux', file='PLOTS/', print=False, save=False):
        if title is None: 
            title = self.title

        if X is not None:
            X = self.wavelength
            y = self.flux

        plt.plot(X, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(title)
        if save: plt.savefig(file + self.target + '_full_spectra.jpg')
        if print: plt.show()

    def plot_full_spectra(self, **kwargs):
        x, y = self.create_spectra(save=True)
        self.plot_spectra(x, y, title='Full Spectra of ' + self.target, **kwargs)


class CalculateParamaters(AnalyzeSpectra):

    def __init__(self, parameters:dict):
        self.parameters = parameters
        
if __name__ == '__main__':
    file = 'HD12871704_15_23_850_20230415064551_47.fit'
    root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
    path = os.path.join(root, file)
    Data = AnalyzeSpectra(root)
    Data.plot_full_spectra(print=True)

    data1 = Data.read_single_file(path)
    Data.plot_spectra(*data1, print=True)