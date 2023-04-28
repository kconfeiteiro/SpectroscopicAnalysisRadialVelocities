from scipy.signal import savgol_filter
from typing import Tuple, Dict, List
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import scipy.constants as const
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
today_time = now.strftime('%Y_%m_%d_%H%M%S')
today = now.strftime('%Y_%m_%d')

class ImportSpectra:

	def __init__(self, path:str, filetype:str='.fit', pattern:str='[A-Za-z]+.[0-9]{5}', smooth_all=False, target:str=None, order:int=None):
		find = lambda ptn, txt: re.findall(ptn, txt)[0]
		ID = lambda x: re.findall('[A-Za-z]+', x)[0] + ' ' + re.findall('[0-9]+', x)[0]
		ffolder = lambda path: os.path.join(*path.split('/')[:2])
		self.smooth_all = smooth_all
		self.pattern = pattern
		self.filetype = filetype
		self.target = target
		self.path = path

		if os.path.isfile(self.path):
			self.order = self.path.split('/')[-1].split('_')[-1].replace('.fit','')
			self.filename = self.path.split('/')[-1].replace('.fit','')
			self.title = self.target + '; Order ' + self.order
			self.folder = ffolder(self.path)
		else:
			self.directory = os.listdir(self.path)
			self.num_files = len(self.directory)
			self.title = 'Spectra of ' + self.target


	def flat(self, x) -> List[float]:
		return [item for subl in x for item in subl]
	
	def smooth_signal(self, y:list, window_size:int=51, poly_order:int=3): 
		return savgol_filter(y, window_size, poly_order)
	
	def read_signal(self, file:str=None, smooth=False) -> Tuple[list, list]:
		if file is None: file = self.path
		sp = fits.open(file)
		header = sp[0].header

		num = header['NAXIS1']
		start_wave, wave_step = header['CRVAL1'], header['CDELT1']
		self.wl = np.linspace(start_wave, start_wave + wave_step*num, num)
		self.wavelength = self.wl.flatten()
		self.flux = sp[0].data
		if smooth or self.smooth_all: self.flux = self.smooth_signal(self.flux)

		return self.wavelength, self.flux
		
	def combine_spectra(self, path:str=None, save_xlsx=False, smooth=False) -> Tuple[list, list]:
		ffolder = lambda path: os.path.join(*path.split('/')[:len(path.split('/') - 1)])
		if (path is None) or (os.path.isdir(path)): path = self.folder
		if os.path.isfile(path): path = ffolder(path)

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
		if smooth or self.smooth_all: self.y = self.smooth_signal(self.y)

		if save_xlsx:
			DATA = pd.DataFrame(data=(self.x, self.y), columns=['Wavelength (Angstroms)','Normalized Flux'])
			DATA.to_excel(self.target.replace(' ','') + '_DATA_' + today)

		return self.x, self.y


class PlotSpectra(ImportSpectra):

	def __init__(self, path:str, save_all=False):
		super().__init__(path)
		self.save_all = save_all

	def plot_spectra(self, X=None, y=None, title=None, xlims:Tuple[float, ...]=None, ylims:tuple=None, 
		  	x_title:str='Wavelength (Angstroms)', y_title:str='Normalized Flux', directory:str='PLOTS/', 
			vlines:List[float]=None, vline_colors:List[str]=None, print=True, save=False, save_as:str=None, 
			figsize:Tuple=(12,7), order:int=None, **kwargs):
		
		if title is None: title = self.title
		if X is None: X, y = self.read_signal(self.path)
		if not os.path.exists(directory): os.mkdir(directory)

		fig, ax = plt.subplots(**kwargs)
		ax.plot(X, y)
		ax.set_title(title if not order else title + f'; Order {order}')
		ax.set_xlabel(x_title)
		ax.set_ylabel(y_title)

		if xlims: ax.set_xlim(*xlims)
		if ylims: ax.set_xlim(*ylims)
		if vlines:
			for line in vlines:
				ax.axvline(line)
		elif vlines and vline_colors:
			for line, color in zip(vlines, vline_colors):
				ax.axvline(line, color=color)

		if os.path.isfile(self.path) and (save or self.save_all): 
			filename = '{}{}_Order-{}_{}.jpg'.format(directory, self.target.replace(' ',''), self.order, today_time)
			fig.savefig(filename)
		elif save_as:
			fig.savefig(directory + save_as)

		if print: plt.show()

	def plot_all(self, wl_order_range:Tuple[int, int], file=None, **kwargs):
		assert len(self.directory) == range(*wl_order_range), f'Range error between \'Directory\' and \'wl_order_range\' of \'{wl_order_range}\''
		for item, order in zip(self.directory, range(*wl_order_range)):
			file2 = os.path.join(self.path, file)
			order = item.split('_')[-1].replace('.fit','')
			data = super().read_signal(file2, smooth=True)
			self.plot_spectra(*data, save_as=file.split('/')[-1].replace('.fit',''), save=True, print=False, order=order, **kwargs)

	def plot_full_spectra(self, data:Tuple[list, list]=None, auto=True, **kwargs):
		if auto: data = self.combine_spectra()
		self.plot_spectra(*data, title='Spectra of ' + self.target, **kwargs)

class CompareSpectra(ImportSpectra):

	def __init__(self, path: str or List = None, filetype:str='.fit'):
		super().__init__(path, filetype)
	
	def parse_folders(self, path: str or List = None) -> List[str]:
		if path is None: path = self.path
		if not os.path.isdir(path):
			raise NotADirectoryError(path + ' is not a directory')
		elif type(self.path) == list and path is None: 
			self.ALL = []
			for folder in path:
				for root, _, files in os.walk(folder):
					for file in files: self.ALL.append(os.path.join(root, file))
			return self.ALL
		else:
			self.ALL = []
			for root, _, files in os.walk(path):
				for file in files: self.ALL.append(os.path.join(root, file))

			return self.ALL
		
	def find_orders(self, order:int, filetype:str=None):
		ID = lambda x: re.findall(f'_[{order}]+' + f'\{self.filetype}' if filetype is None else f'_[{order}]+' + f'\{self.filetype}', x)
		return [file for file in os.listdir(self.path) if ID(file)]
	
	def plot_comparison(self, x_label=None, y_label=None, files:List=None, save_as=None, xlims=None, ylims=None, vlines=None, vline_colors=None, show=False, **kwargs):
		if files is None:
			fig, ax = plt.subplots(**kwargs)
			for file in self.ALL:
				ax.plot(*super().read_signal(file))
			
			if x_label: ax.set_xlabel(x_label)
			if y_label: ax.set_ylabel(y_label)

			if xlims: ax.set_xlim(*xlims)
			if ylims: ax.set_xlim(*ylims)
			if vlines:
				for line in vlines:
					ax.axvline(line)
			elif vlines and vline_colors:
				for line, color in zip(vlines, vline_colors):
					ax.axvline(line, color=color)

			if save_as: fig.savefig(save_as)
			if show: plt.show()


class AnalyzeSpectra(ImportSpectra):

	def __init__(self, path:str, target_wavelength:int=6560):
		super().__init__(path)
		self.target_wavelength = target_wavelength

	def normalize(self, lst:list) -> List[float]:
		return [(value - (min(lst))/max(lst)) for value in np.array(lst)]

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
	
	def calculate_dlambda(self, target_wavelength: float or int = None, df1:list=None, df2: list = None, index: float or int = None):
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