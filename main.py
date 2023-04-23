from ProjectMain.Package import AnalyzeSpectra, ImportSpectra, PlotSpectra
import os

if __name__ == '__main__':
    file = 'HD12871704_15_23_850_20230415064551_37.fit'
    root = 'DATA/HD12871704_15_23_850_-1_20230415T064551/'
    path = os.path.join(root, file)
    # Data = AnalyzeSpectra(root)
    # Data.plot_full_spectra(print=True)

    # pre = ImportSpectra(path)
    # x, y = pre.read_raw_signal()

    DATA = PlotSpectra(path)
    DATA.read_signal()
    DATA.plot_spectra(print=True)
    DATA.combine_spectra(root)
    DATA.plot_full_spectra()