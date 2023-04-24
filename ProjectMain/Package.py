from specutils import SpectralRegion, Spectrum1D
from astropy.nddata import StdDevUncertainty
from specutils.analysis import correlation
from astropy.modeling import models
import matplotlib.pyplot as plt
import scipy.constants as const
from astropy import units as u
import scipy.signal as signal
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
import numpy as np
import math as m
import os


'''
Links (sources)
-----
1. https://astronomy.stackexchange.com/questions/43552/working-with-stellar-spectra-in-fits-format-in-python

'''

class ImportSpectra:

    def __init__(self, path:str):
        self.path = path

        if os.path.isfile(path): 
            self.filename = self.path.split('/')[-1].replace('.fit','')
            self.target = self.filename.split('_')[0]
            self.title = self.target + '; Order ' + self.filename.split('_')[-1]
        else:
            self.directory = os.listdir(self.path)
            self.target = self.directory[0].split('/')[-1].replace('.fit','').split('_')[0]
            self.title = 'Spectra of ' + self.target

    def flatten(x):
        return [item for subl in x for item in subl]

    def read_signal(self, file=None) -> tuple:
        if file is None: file = self.path
        sp = fits.open(file)
        header = sp[0].header

        num = header['NAXIS1']
        start_wave=header['CRVAL1']
        wave_step=header['CDELT1']
        self.wl = np.linspace(start_wave, start_wave + wave_step*num, num)

        self.wavelength = self.wl.flatten()
        self.flux = sp[0].data

        return self.wavelength, self.flux
        
    def combine_spectra(self, path=None):
        if path is None: path = self.path

        self.x, self.y = [0]*len(os.listdir(path)), [0]*len(os.listdir(path))
        for i, file in enumerate(os.listdir(path)):
            file = os.path.join(path, file)
            self.x[i], self.y[i] = self.read_signal(file)

        self.x, self.y = self.flatten(self.x), self.flatten(self.y)
        assert type(self.x) == list, type(self.y) == list
        return self.x, self.y


class PlotSpectra(ImportSpectra):

    def __init__(self, path:str):
        super().__init__(path)

    def plot_spectra(self, X=None, y=None, title=None, x_title='Wavelength (Angstroms)', y_title='Normalized Flux', file='PLOTS/', print=False, save=False):
        if title is None: title = self.title
        if X is None: X, y = self.wavelength, self.flux

        plt.plot(X, y)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(title)
        plt.tight_layout()
        if save: plt.savefig(file + self.target + '_full_spectra.jpg')
        if print: plt.show()

    def plot_full_spectra(self, manual=False, **kwargs):
        if not manual: x, y = self.combine_spectra(save=True)
        self.plot_spectra(x, y, title='Full Spectra of ' + self.target, **kwargs)


class AnalyzeSpectra(ImportSpectra):

    def __init__(self, path:str, parameters:dict):
        super().__init__(path)
        self.parameters = parameters

    def smooth_signal(self, y):
        return np.linalg.svd(y, full_matrices=False)    
    
    def fit_period(self, X=None, y=None, setup=(0.01, 10, 100000), **kwargs):
        if X is None: X, y = self.wavelength, self.flux
        self.w = np.linspace(*setup)
        self.PDGM = signal.lombscargle(X, y, setup[-1], **kwargs)
        return self.PDGM, self.w
    
    def show_periodogram(self, show=True, **kwargs):
        _, (og, pdgm) = plt.subplots(2, 1, constrained_layout=True, **kwargs)
        og.plot(self.wavelength, self.flux)
        og.set_xlabel('Time [s]')
        pdgm.plot(self.PDGM, self.w)
        pdgm.set_xlabel('Angular Frequency [rad/s]')
        pdgm.set_ylabel('Normalized Amplitude')
        if show: plt.show()
    
    def calculate_dlambda(self, target_wavelength=None, df1=None, df2=None, index=None):
        assert len(df1) == len(df2)
        if target_wavelength is None: target_wavelength = 6560 

        calc = lambda x: m.log(x)
        c1, c2 = list(map(calc, np.array(df1))), list(map(calc, np.array(df2)))
        diff = np.subtract(c2, c1)
        if index is None: 
            return diff
        else:
            target_wavelength = self.target_wavelength
            self.new = np.stack((self.wl, c1, c2), axis=1)
            return self.new[target_wavelength,:]
       
    def calculate_radial_velocity(self, delta_lambda=None, lam=None):
        if delta_lambda is None:
            delta_lambda = self.new[self.target_wavelength,:]
            lam = 6560
        return const.c*(delta_lambda/lam)