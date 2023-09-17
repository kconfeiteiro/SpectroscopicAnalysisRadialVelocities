from ProjectMain.Package import (
	AnalyzeSpectra, ImportSpectra,
	PlotSpectra, CompareSpectra
)
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os


[print("Hello world") for _ in range(100)]
[print("Hello world") for _ in range(100)]

exit()
if __name__ == '__main__':

	file = 'HD12871704_15_23_850_20230415064551_34.fit'
	root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
	path = os.path.join(root, file)

	MAIN = PlotSpectra(path, order=37, filetype='.fit', target='HD 128717', smooth_all=True)
	# MAIN.plot_spectra(x_title='Wavelength (Angstroms)', y_title='Flux', tight_layout=True, show_lowest=3)

	ANT_root = 'DATA/Antonio Spectra/HD 46150 (4.7.23) Full Spectra'
	file1 = 'DATA/Antonio Spectra/HD 46150 (4.7.23) Full Spectra/HD 46150 (4.7.23)_20230408012534_46.fit'
	PTH1 = 'DATA/Antonio Spectra/HD 46150 (4.7.23) Full Spectra/'
	file2 = 'DATA/Antonio Spectra/HD 46150 (4.20.23)_-1_20230421T012334/HD 46150 (4.20.23)_20230421012334_46.fit'

	ANT1 = PlotSpectra('DATA/Antonio Spectra/HD 50896 Full Spectra', filetype='.fit', target='HD 46150')
	full1 = ANT1.combine_spectra(smooth=False)
	full2 = ANT1.combine_spectra(smooth=True)

	plt.plot(*full1, *full2)
	plt.xlabel('Wavelength (Angstroms)')
	plt.ylabel('Flux')
	plt.title('HD 46150 Full Spectra (smoothed vs. raw)')
	plt.legend(['Raw signal','Smoothed Signal'])
	plt.savefig('HD 46150 Full Spectra (smoothed vs. raw).jpg', dpi=500)
	plt.show()

	# ANT2 = PlotSpectra(file2)
	# data2 = ANT2.read_signal(smooth=True)

	# file3 = 'DATA\Antonio Spectra/HD 50896 Full Spectra/HD 50896 (new).obs.srl_20230321005851_46.fit'
	# ANT3 = PlotSpectra(file3)
	# data3 = ANT3.read_signal(smooth=True)

	# plt.plot(*data1, *data2, *data3)
	# plt.xlabel('Wavelength (Angstroms)')
	# plt.ylabel('Flux')
	# plt.axvline(4861)
	# plt.title('Antonion + Evan')
	# plt.legend(['04/07','04/20','03/21 (diff. Target)'])

	# FOLDERS = [
		# 'DATA/Antonio Spectra/HD 46150 (4.7.23) Full Spectra/',
		# 'DATA/Antonio Spectra/HD 46150 (4.20.23)_-1_20230421T012334',
		# 'DATA/Antonio Spectra/HD 46150 (old-bad) Full Spectra'
	# ]

	# COMP = CompareSpectra(FOLDERS)
	# files = COMP.parse_folders()
	# print(files)
	# needed = COMP.find_orders(order=43, filetype='.fit')
	# COMP.plot_comparison(x_label='Wavelength', y_label='Flux')
