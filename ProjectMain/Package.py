from specutils import SpectralRegion, Spectrum1D
from astropy.nddata import StdDevUncertainty
from specutils.analysis import correlation
from typing import Tuple, Dict, List
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
2. py -m pip install git+https://github.com/timsainb/noisereduce (error when installing); import noisereduce as nr

'''

now = dt.datetime.now()
today_time = now.strftime('%Y_%m_%d_%H%M%S')
today = now.strftime('%Y_%m_%d')

class ImportSpectra:

	def __init__(self, path:str, pattern:str='[A-Za-z]+.[0-9]{5}'):
		find = lambda ptn, txt: re.findall(ptn, txt)[0]
		ID = lambda x: re.findall('[A-Za-z]+', x)[0] + ' ' + re.findall('[0-9]+', x)[0]
		ffolder = lambda path: os.path.join(*path.split('/')[:2])
		self.pattern = pattern

		self.path = path
		targ = find(self.pattern, self.path)
		self.target = ID(targ)

		if os.path.isfile(path): 
			self.order = self.path.split('/')[-1].split('_')[-1].replace('.fit','')
			self.filename = self.path.split('/')[-1].replace('.fit','')
			self.title = self.target + '; Order ' + self.order
			self.folder = ffolder(self.path)
		else:
			self.directory = os.listdir(self.path)
			self.num_files = len(self.directory)
			self.title = 'Spectra of ' + self.target

	def flat(self, x) -> list:
		return [item for subl in x for item in subl]
	
	def smooth_signal(self, y:list):
		return np.linalg.svd(np.array(y), full_matrices=False) 
	
	def read_signal(self, file:str=None) -> Tuple:
		if file is None: file = self.path
		sp = fits.open(file)
		header = sp[0].header

		num = header['NAXIS1']
		start_wave, wave_step = header['CRVAL1'], header['CDELT1']
		self.wl = np.linspace(start_wave, start_wave + wave_step*num, num)
		self.wavelength = self.wl.flatten()
		self.flux = sp[0].data

		return self.wavelength, self.flux
		
	def combine_spectra(self, path:str=None, save_xlsx=False) -> Tuple[list, list]:
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

		if save_xlsx:
			DATA = pd.DataFrame(data=(self.x, self.y), columns=['Wavelength (Angstroms)','Normalized Flux'])
			DATA.to_excel(self.target.replace(' ','') + '_DATA_' + today)

		return self.x, self.y


class PlotSpectra(ImportSpectra):

	def __init__(self, path:str, save_all=False):
		super().__init__(path)
		self.save_all = save_all

	def plot_spectra(self, X=None, y=None, title=None, xlims:tuple=None, ylims:tuple=None, x_title:str='Wavelength (Angstroms)', y_title:str='Normalized Flux', directory:str='PLOTS/SingleOrder/', vlines:list[float]=None, vline_colors:list[str]=None, print=True, save=False, vline=None, save_as=None, **kwargs):
		
		if title is None: title = self.title
		if (X and y) is None: X, y = self.read_signal(self.path)


		plt.plot(X, y, **kwargs)

		plt.title(title)
		plt.xlabel(x_title)
		plt.ylabel(y_title)

		if xlims: plt.xlim(*xlims)
		if ylims: plt.xlim(*ylims)
		if vline: plt.axvline(vlines, vline_colors)

		plt.tight_layout()
		if save or self.save_all: 
			if not os.path.exists(directory): os.mkdir(directory)
			filename = '{}{}_Order-{}_{}.jpg'.format(directory, self.target.replace(' ',''), self.order, today_time)
			plt.savefig(filename)
			# print(f'Your plot has been saved in \'{directory}\'')
		elif save_as:
			plt.savefig(save_as)
			print(f'Your plot has been saved as {save_as}')

		if print: plt.show()

	def plot_full_spectra(self, data=None, auto=True, **kwargs):
		if auto: data = self.combine_spectra()
		self.plot_spectra(*data, title='Spectra of ' + self.target, **kwargs)


class AnalyzeSpectra(ImportSpectra):

	def __init__(self, path:str, target_wavelength:int=6560):
		super().__init__(path)
		self.target_wavelength = target_wavelength
	

	def reduce_noise(self) -> Tuple[list, list]:
		pass # placeholder

	def fit_period(self, X=None, y=None, setup:Tuple=(0.01, 10, 100000), **kwargs) -> Tuple[list, list]:
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
		
		if save or self.save_all: fig.savefig(self.target.replace(' ','') + '_PeriodogramAnalysis_' + today_time)
		if show: plt.show()
	
	def calculate_dlambda(self, target_wavelength: float | int = None, df1:list=None, df2: list = None, index: float | int = None):
		if target_wavelength is None: target_wavelength = self.target_wavelength
		assert len(df1) == len(df2)

		calc = lambda x: m.log(x)
		c1, c2 = list(map(calc, df1)), list(map(calc, df2))
		diff = np.subtract(c2, c1)
		diff = np.stack((self.w))

		if index is None: 
			return diff
		else:
			self.new = np.stack((self.wl, c1, c2), axis=1)
			return self.new[self.target_wavelength,:]
	   
	def calculate_radial_velocity(self, delta_lambda=None, lam=6563):
		if delta_lambda is None: delta_lambda = self.new[self.target_wavelength,:]
		return const.c*(delta_lambda/lam)


class Report(ImportSpectra):

	def __init__(self, path:str, save_to=f'{today}/REPORT/'):
		super().__init__(path)
		self.save_to = save_to

	def summary(self):
		print(f'''
			FILE: \'{self.filename}\'
			PATH: \'{self.path}\'
			TARGET: \'{self.target}\'
			NUM. FILES: \'{self.num_files}\'
		''')

	def write_to_text(self, filename: str = None, lines: list or str = None):

		with open(f'{self.save_to}{filename}', 'w') as file:
			if type(lines) == str:
				file.write(lines)
			elif type(lines) == list:
				for line in lines:
					line = str(line)
					file.write(line + '\n')

		return print(f'Your file has been saved as \'{filename}\'')