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
import datetime as dt
import pandas as pd
import numpy as np
import math as m
import os, re


'''
Links (sources)
-----
1. https://astronomy.stackexchange.com/questions/43552/working-with-stellar-spectra-in-fits-format-in-python

'''

now = dt.datetime.now()
now = now.strftime('%Y_%m_%d_%H%M%S')

class ImportSpectra:

    def __init__(self, path:str, pattern:str='[A-Za-z]+.[0-9]{5}'):
        find = lambda ptn, txt: re.findall(ptn, txt)[0]
        ID = lambda x: re.findall('[A-Za-z]+', x)[0] + ' ' + re.findall('[0-9]+', x)[0]
        ffolder = lambda path: os.path.join(*path.split('/')[:2])

        self.path = path
        targ = find('[A-Za-z]+.[0-9]{5}', self.path)
        self.target = ID(targ)
        self.pattern = pattern

        if os.path.isfile(path): 
            self.filename = self.path.split('/')[-1].replace('.fit','')
            self.title = self.target + '; Order ' + self.target
            self.folder = ffolder(self.path)
        else:
            self.directory = os.listdir(self.path)
            self.title = 'Spectra of ' + self.target

    def flat(self, x) -> list:
        return [item for subl in x for item in subl]
    
    def smooth_signal(self, y:list):
        return np.linalg.svd(np.array(y), full_matrices=False) 
    
    def read_signal(self, file:str=None) -> tuple:
        if file is None: file = self.path
        sp = fits.open(file)
        header = sp[0].header

        num = header['NAXIS1']
        start_wave, wave_step = header['CRVAL1'], header['CDELT1']
        self.wl = np.linspace(start_wave, start_wave + wave_step*num, num)
        self.wavelength = self.wl.flatten()
        self.flux = sp[0].data

        return self.wavelength, self.flux
        
    def combine_spectra(self, path:str=None) -> tuple:
        if (path is None) or (os.path.isdir(path)): path = self.folder
        if os.path.isfile(path): 
            ffolder = lambda path: os.path.join(*path.split('/')[:len(path.split('/')-1)])
            path = ffolder(path)

        assert os.path.isdir(path), f'Path is not a directory. Could not parse \'{path}\''
        self.x, self.y = [0]*len(os.listdir(path)), [0]*len(os.listdir(path))
        for i, file in enumerate(os.listdir(path)):
            file = os.path.join(path, file)
            self.x[i], self.y[i] = self.read_signal(file)

        self.x, self.y = self.flat(self.x), self.flat(self.y)
        assert type(self.x) == type(self.y) == list

        self.data = np.stack((self.x, self.y), axis=1)
        self.data = self.data[self.data[:,0].argsort()]
        self.x, self.y = self.data[:,0], self.data[:,1]

        return self.x, self.y


class PlotSpectra(ImportSpectra):

    def __init__(self, path:str, pattern:str='[A-Za-z]+.[0-9]{5}'):
        super().__init__(path, pattern)

    def plot_spectra(self, X=None, y=None, title=None, xlims:tuple=None, ylims:tuple=None,
            x_title:str='Wavelength (Angstroms)', y_title:str='Normalized Flux',
            file:str='PLOTS/', print=True, save=False):
        
        if title is None: title = self.title
        plt.plot(X, y) #, s=2)
        plt.xlabel(x_title)

        if xlims: plt.xlim(*xlims)
        if ylims: plt.ylim(*ylims)
        plt.ylabel(y_title)
        plt.title(title)
        plt.tight_layout()

        if save: plt.savefig(file + self.target + '_full_spectra_' + now +'.jpg')
        if print: plt.show()

    def plot_full_spectra(self, auto=True, **kwargs):
        if auto: data = self.combine_spectra()
        self.plot_spectra(*data, title='Spectra of ' + self.target, **kwargs)


class AnalyzeSpectra(ImportSpectra):

    def __init__(self, path:str, parameters:dict, target_wavelength:int=6560, pattern:str='[A-Za-z]+.[0-9]{5}'):
        super().__init__(path, pattern)
        self.parameters = parameters
        self.target_wavelength = target_wavelength
    
    def fit_period(self, X=None, y=None, setup=(0.01, 10, 100000), **kwargs):
        if X is None: X, y = self.wavelength, self.flux
        self.w = np.linspace(*setup)
        self.PDGM = signal.lombscargle(X, y, setup[-1], **kwargs)
        return self.PDGM, self.w
    
    def show_periodogram(self, show=True, save=False, **kwargs):
        fig, (og, pdgm) = plt.subplots(2, 1, constrained_layout=True, **kwargs)

        og.plot(self.wavelength, self.flux)
        og.set_xlabel('Time [s]')

        pdgm.plot(self.PDGM, self.w)
        pdgm.set_xlabel('Angular Frequency [rad/s]')
        pdgm.set_ylabel('Normalized Amplitude')
        fig.set_label(self.target + ' Periodogram Analysis')
        
        if save: fig.savefig(self.target.replace(' ','') + '_PeriodogramAnalysis_' + now)
        if show: plt.show()
    
    def calculate_dlambda(self, target_wavelength=None, df1=None, df2=None, index=None):
        if target_wavelength is None: target_wavelength = self.target_wavelength
        assert len(df1) == len(df2)

        calc = lambda x: m.log(x)
        c1, c2 = list(map(calc, np.array(df1))), list(map(calc, np.array(df2)))
        diff = np.subtract(c2, c1)
        diff = np.stack((self.w))

        if index is None: 
            return diff
        else:
            target_wavelength = self.target_wavelength
            self.new = np.stack((self.wl, c1, c2), axis=1)
            return self.new[target_wavelength,:]
       
    def calculate_radial_velocity(self, delta_lambda=None, lam=6560):
        if delta_lambda is None:
            delta_lambda = self.new[self.target_wavelength,:]
        return const.c*(delta_lambda/lam)
    
    def summary(self):
        print(f'''
        FILE:   {self.filename}
        PATH:   {self.path}
        TARGET: {self.target}
        ''')