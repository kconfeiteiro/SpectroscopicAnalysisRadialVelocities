import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from specutils import SpectralRegion, Spectrum1D
from specutils.analysis import correlation

dir='/home/user/EMILYS_FITS_DATA/Beta_Virginis_NightA_5/'
file_list = ['34.fit']

for i in range(0, np.size(file_list)):    

    df=fits.open(dir+file_list[i])
    data = df[0].data
    hdr = df[0].header

    #hdr
    start_wave=hdr['CRVAL1']
    wave_step=hdr['CDELT1']
    num_wave_points=hdr['NAXIS1']
    end_wave=start_wave + wave_step*num_wave_points
    wavelengths=np.linspace(start_wave, end_wave, num_wave_points)

    fig, ax = plt.subplots()
    plt.style.use('seaborn-whitegrid')
    ax.plot(wavelengths, data)
    ax.set_xlabel("Wavelengths [Angstroms]", fontsize = 23)
    ax.set_title("Beta Virginis Night A Order 34 5-Min ExpTime", fontsize = 28)
    ax.set_ylabel("Relative Intensity [counts]", fontsize = 23)


dir1 = '/home/user/EMILYS_FITS_DATA/Beta_Virginis_NightA_5/'
file1 = '35.fit'
dir2 = '/home/user/EMILYS_FITS_DATA/testfolder/'
file2 = 'Newobservation_20230416034739_35.fit'
    
df=fits.open(dir1 + file1)
data1 = df[0].data
hdr = df[0].header
print(hdr)

start_wave = hdr['CRVAL1']
wave_step = hdr['CDELT1']
num_wave_points = hdr['NAXIS1']
end_wave = start_wave + wave_step*num_wave_points
wavelengths1 = np.linspace(start_wave, end_wave, num_wave_points)

df = fits.open(dir2+file2)
data2 = df[0].data
hdr = df[0].header

start_wave=hdr['CRVAL1']
wave_step=hdr['CDELT1']
num_wave_points=hdr['NAXIS1']
end_wave=start_wave + wave_step*num_wave_points
wavelengths2=np.linspace(start_wave, end_wave, num_wave_points)
    
""" THIS IS WHERE NEW CODE BEGINS """
spectral_axis1 = wavelengths1*u.AA
spec_flux1=data1*u.Jy

spectral_axis2 = wavelengths2*u.AA
spec_flux2=data2*u.Jy

np.random.seed(42)
spectral_axis = np.linspace(11., 1., 200) * u.GHz

spectral_model = models.Gaussian1D(amplitude=5*(2*np.pi*0.8**2)**-0.5*u.Jy, mean=5*u.GHz, stddev=0.8*u.GHz)
flux = spectral_model(spectral_axis)
flux += np.random.normal(0., 0.05, spectral_axis.shape) * u.Jy
uncertainty = StdDevUncertainty(0.2*np.ones(flux.shape)*u.Jy)
noisy_gaussian = Spectrum1D(spectral_axis=spectral_axis, flux=flux, uncertainty=uncertainty)

size = 200
spec_axis = np.linspace(4500., 6500., num=size) * u.AA

rest_value = 6000. * u.AA   #? How did they get these values? 

# Uncertainty
uncertainty = StdDevUncertainty(0.2*np.ones(size)*u.Jy)

# 1
def extract_spec(data1, spectral_axis1, spec_flux1, uncertainty, size, spec_axis, rest_value):
    mean1 = 5035. * u.AA #? How did they get these values? 
    f1 = np.random.randn(size)*0.5 * u.Jy
    g1 = models.Gaussian1D(amplitude=30*u.Jy, mean=mean1, stddev=10. * u.AA)
    flux1 = f1 + g1(spec_axis)
    spec_size1 = np.size(data1)
    spec_uncertainty1 = StdDevUncertainty(0.02*np.ones(spec_size1)*u.Jy)
    ospec = Spectrum1D(spectral_axis=spec_axis, flux=flux1, uncertainty=uncertainty, velocity_convention='optical', rest_value=rest_value)
    ospec1 = Spectrum1D(spectral_axis=spectral_axis1, flux=spec_flux1, uncertainty=spec_uncertainty1, velocity_convention='optical', rest_value=rest_value)

extract_spec(data1, spectral_axis1, spec_flux1, uncertainty, size, spec_axis, rest_value)
mean1 = 5035. * u.AA #? How did they get these values? 
f1 = np.random.randn(size)*0.5 * u.Jy
g1 = models.Gaussian1D(amplitude=30*u.Jy, mean=mean1, stddev=10. * u.AA)
flux1 = f1 + g1(spec_axis)
spec_size1 = np.size(data1)
spec_uncertainty1 = StdDevUncertainty(0.02*np.ones(spec_size1)*u.Jy)
ospec = Spectrum1D(spectral_axis=spec_axis, flux=flux1, uncertainty=uncertainty, velocity_convention='optical', rest_value=rest_value)
ospec1 = Spectrum1D(spectral_axis=spectral_axis1, flux=spec_flux1, uncertainty=spec_uncertainty1, velocity_convention='optical', rest_value=rest_value)

# 2
mean2 = 5015. * u.AA #? How did they get these values? 
f2 = np.random.randn(size)*0.5 * u.Jy
g2 = models.Gaussian1D(amplitude=30*u.Jy, mean=mean2, stddev=10. * u.AA)
flux2 = f2 + g2(spec_axis)
spec_size2 = np.size(data2)
spec_uncertainty2 = StdDevUncertainty(0.02*np.ones(spec_size2)*u.Jy)
tspec = Spectrum1D(spectral_axis=spec_axis, flux=flux2, uncertainty=uncertainty)
tspec2 = Spectrum1D(spectral_axis=spectral_axis2, flux=spec_flux2, uncertainty=spec_uncertainty2)

# cross corrleatoin
corr, lag = correlation.template_correlate(tspec, tspec2)
corr, lag = correlation.template_correlate(ospec, tspec)

# ax.plot(lag, corr)
# ax.set_xlim(-10000, 10000)
# ax.set_xlabel("Lag", fontsize = 20)
# ax.set_title("Correlation example", fontsize = 20)
# ax.set_ylabel("Correlation", fontsize = 20)

# FOR Example, the shift should be (20/5015)*3
expected=3e5*(20/5015)
print('expected lag: ', expected)
ax.plot([expected,expected],[0,800])

expected=3e5*(0/5015)
ax.plot(lag, corr)