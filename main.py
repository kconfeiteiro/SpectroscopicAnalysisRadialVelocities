from ProjectMain.Package import AnalyzeSpectra, ImportSpectra, PlotSpectra
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    file = 'HD12871704_15_23_850_20230415064551_37.fit'
    root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
    path = os.path.join(root, file)
    # Data = AnalyzeSpectra(root)
    # Data.plot_full_spectra(print=True)

    pre = PlotSpectra(path)
    xy = pre.read_signal()
    pre.plot_spectra(*xy, print=False)

    DATA = PlotSpectra(path)
    dat = DATA.read_signal()
    DATA.plot_spectra(*dat, print=False)
    DATA.combine_spectra(root)
    DATA.plot_full_spectra(save=True, c='Brown')
    
    antonion = 'DATA\Antonio Spectra/HD 46150 (4.7.23) Full Spectra'
    FILE = 'HD 46150 (4.7.23)_20230408012534_36.fit'
    pth = os.path.join(antonion, FILE)
    ANT = PlotSpectra(pth)
    data = ANT.read_signal()
    ANT.plot_spectra(*data, xlims=(6200,6300), print=False)