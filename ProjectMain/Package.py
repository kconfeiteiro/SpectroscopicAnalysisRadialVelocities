from typing import Tuple, List, Sequence  # , Dict
from scipy.signal import savgol_filter
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

"""
Links (sources)
-----
1. https://astronomy.stackexchange.com/questions/43552/working-with-stellar-spectra-in-fits-format-in-python

"""

now = dt.datetime.now()
today_time = now.strftime("%Y_%m_%d_%H%M%S")
today = now.strftime("%Y_%m_%d")


class ImportSpectra:
    def __init__(
        self,
        path: str,
        filetype: str = ".fit",
        smooth_all=False,
        target: str = None,
        order: int = None,
    ):  # FIXME - reorganzie this constructor
        find_folder = lambda path: os.path.join(*path.split("/")[:2])
        self.smooth_all = smooth_all
        self.filetype = filetype
        self.target = target
        self.order = order
        self.path = path
        self.pattern = f"_[{self.order}]+" + f"\{self.filetype}"

        if os.path.isfile(self.path):
            self.filename = self.path.split("/")[-1].replace(".fit", "")
            self.title = self.target + "; Order " + str(self.order)
            self.folder = find_folder(self.path)
        else:
            self.directory = os.listdir(self.path)
            self.num_files = len(self.directory)
            self.title = "Spectra of " + self.target

    def flat(self, x) -> List[float]:
        return [item for subl in x for item in subl]

    def find_n_lowest(
        self, array: Sequence, num_of_values: int = 1, find_unique=False
    ) -> Sequence[float]:
        array = sorted(array)
        if find_unique:
            array = np.unique(sorted(array))
        return array[0:num_of_values]

    def smooth_signal(
        self, y: list, window_size: int = 51, poly_order: int = 3, **kwargs
    ):
        return savgol_filter(y, window_size, poly_order, **kwargs)

    def read_signal(self, file: str = None, smooth=False) -> Tuple[list, list]:
        if file is None:
            file = self.path
        sp = fits.open(file)
        header = sp[0].header

        num = header["NAXIS1"]
        start_wave, wave_step = header["CRVAL1"], header["CDELT1"]
        self.wl = np.linspace(start_wave, start_wave + wave_step * num, num)
        self.wavelength = self.wl.flatten()
        self.flux = sp[0].data
        if smooth or self.smooth_all:
            self.flux = self.smooth_signal(self.flux)

        return self.wavelength, self.flux

    def combine_spectra(
        self, path: str = None, save_xlsx_as=None, smooth=False
    ) -> Tuple[list, list]:
        ffolder = lambda path: os.path.join(
            *path.split("/")[: len(path.split("/") - 1)]
        )
        if (path is None) or (os.path.isdir(path)):
            path = self.path
        elif os.path.isfile(path):
            path = ffolder(path)

        assert os.path.isdir(path), f"Path is not a directory. Could not parse '{path}'"
        self.x, self.y = [0] * len(os.listdir(path)), [0] * len(os.listdir(path))
        for i, file in enumerate(os.listdir(path)):
            file = os.path.join(path, file)
            self.x[i], self.y[i] = self.read_signal(file)

        self.x, self.y = self.flat(self.x), self.flat(self.y)
        assert type(self.x) == type(self.y) == list

        self.data = np.stack((self.x, self.y), axis=1)
        self.data = self.data[self.data[:, 0].argsort()]
        self.x, self.y = self.data[:, 0], self.data[:, 1]
        if smooth or self.smooth_all:
            self.y = self.smooth_signal(self.y)

        if save_xlsx_as:
            DATA = pd.DataFrame(
                data=(self.x, self.y),
                columns=["Wavelength (Angstroms)", "Normalized Flux"],
            )
            DATA.to_excel(self.target.replace(" ", "") + "_DATA_" + today)

        return self.x, self.y


class PlotSpectra(ImportSpectra):
    def __init__(
        self,
        path: str,
        order=None,
        save_all=False,
        filetype: str = ".fit",
        target: str = None,
        **kwargs,
    ):
        super().__init__(path, filetype=filetype, order=order, target=target, **kwargs)
        self.save_all = save_all

    def plot_spectra(
        self,
        X=None,
        y=None,
        title=None,
        xlims: Tuple[float, float] = None,
        ylims: Tuple[float, float] = None,
        x_title: str = None,
        y_title: str = None,
        directory: str = "PLOTS/",
        vlines: List[float] = None,
        vline_colors: List[str] = None,
        print=True,
        save=False,
        save_as: str = None,
        show_lowest: int = None,
        order: int = None,
        annotations: Sequence[str] = None,
        tight_layout=False,
        **kwargs,
    ):
        if title is None:
            title = self.title
        if (X is None) and os.path.isfile(self.path):
            X, y = self.read_signal(self.path)
        if order is None:
            order = self.order
        if not os.path.exists(directory):
            os.mkdir(directory)

        fig, ax = plt.subplots(**kwargs)
        ax.plot(X, y)
        ax.set_title(title)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        y_limits = ax.get_ylim()

        if xlims:
            ax.set_xlim(*xlims)
        if ylims:
            ax.set_xlim(*ylims)

        if show_lowest:  # FIXME - this does not wokr at all! :(
            limits = super().find_n_lowest(y, show_lowest)
            for line in limits:
                ax.axvline(line, y_limits[0], y_limits[1])
                ax.annotate(str(line), xy=(line + 0.05, max(y_limits) * 0.75))

        if vlines:
            for line in vlines:
                ax.axvline(line)
        elif vlines and vline_colors:
            for line, color, ann in zip(vlines, vline_colors, annotations):
                ax.axvline(line, color=color)
                if ann:
                    ax.annotate(ann, xy=(line + 0.05, 5.5), color="green")

        if tight_layout:
            fig.tight_layout()
        if os.path.isfile(self.path) and (save or self.save_all):
            fig.savefig(
                "{}{}_Order-{}_{}.jpg".format(
                    directory, self.target.replace(" ", ""), self.order, today_time
                )
            )
        elif save_as:
            fig.savefig(os.path.join(directory, save_as))

        if print:
            plt.show()

    def plot_all(self, wl_order_range: Tuple[int, int], **kwargs):
        assert len(self.directory) == range(
            *wl_order_range
        ), f"Range error between 'Directory' and 'wl_order_range' of '{wl_order_range}'"
        for item, order in zip(self.directory, range(*wl_order_range)):
            file2 = os.path.join(self.path, item)
            order = item.split("_")[-1].replace(".fit", "")
            data = super().read_signal(file2, smooth=True)
            self.plot_spectra(
                *data,
                save_as=item.split("/")[-1].replace(".fit", ""),
                save=True,
                print=False,
                order=order,
                **kwargs,
            )

    def plot_full_spectra(self, data: Tuple[list, list] = None, auto=True, **kwargs):
        if auto:
            data = self.combine_spectra()
        self.plot_spectra(*data, title="Spectra of " + self.target, **kwargs)


class CompareSpectra(ImportSpectra):
    def __init__(self, path: str or List = None, filetype: str = ".fit"):
        super().__init__(path, filetype)

    def parse_folders(self, path: str or List = None) -> List[str]:
        if path is None:
            path = self.path
        if type(self.path) == list:
            self.ALL = [
                os.path.join(root, file)
                for folder in path
                for root, _, files in os.walk(folder)
                for file in files
            ]
            return self.ALL
        else:
            self.ALL = [
                os.path.join(root, file)
                for root, _, files in os.walk(path)
                for file in files
            ]
            return self.ALL

    def find_orders(self, order: int, filetype: str = None):
        ID = lambda x: re.findall(
            f"_[{order}]+" + f"\{self.filetype}"
            if filetype is None
            else f"_[{order}]+" + f"\{self.filetype}",
            x,
        )
        return [file for file in os.listdir(self.path) if ID(file)]

    def plot_comparison(
        self,
        x_label=None,
        y_label=None,
        files: List = None,
        save_as=None,
        xlims=None,
        ylims=None,
        vlines=None,
        vline_colors=None,
        show=False,
        **kwargs,
    ):
        if files is None:
            fig, ax = plt.subplots(**kwargs)
            for file in self.ALL:
                ax.plot(*super().read_signal(file))

            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)

            if xlims:
                ax.set_xlim(*xlims)
            if ylims:
                ax.set_xlim(*ylims)
            if vlines:
                for line in vlines:
                    ax.axvline(line)
            elif vlines and vline_colors:
                for line, color in zip(vlines, vline_colors):
                    ax.axvline(line, color=color)

            if save_as:
                fig.savefig(save_as)
            if show:
                plt.show()


class AnalyzeSpectra(ImportSpectra):
    def __init__(self, path: str, target_wavelength: int = 6560):
        super().__init__(path)
        self.target_wavelength = target_wavelength

    def normalize(self, lst: list) -> List[float]:
        return [(value - (min(lst)) / max(lst)) for value in np.array(lst)]

    def fit_period(
        self, X=None, y=None, setup: Tuple = (0.01, 10, 100000), **kwargs
    ) -> Tuple[list, list]:
        if X is None:
            X, y = self.wavelength, self.flux
        self.w = np.linspace(*setup)
        self.PDGM = signal.lombscargle(X, y, setup[-1], **kwargs)
        return self.PDGM, self.w

    def show_periodogram(self, show=True, save=False, orders=(2, 1), **kwargs):
        fig, (og, pdgm) = plt.subplots(*orders, constrained_layout=True, **kwargs)

        og.plot(self.wavelength, self.flux)
        og.set_xlabel("Time [s]")

        pdgm.plot(self.PDGM, self.w)
        pdgm.set_xlabel("Angular Frequency [rad/s]")
        pdgm.set_ylabel("Normalized Amplitude")
        fig.set_label(self.target + " Periodogram Analysis")

        if save or self.save_all:
            fig.savefig(
                self.target.replace(" ", "") + "_PeriodogramAnalysis_" + today_time
            )
        if show:
            plt.show()

    def calculate_dlambda(
        self,
        target_wavelength: float or int = None,
        df1: List = None,
        df2: List = None,
        index: float or int = None,
    ):
        if target_wavelength is None:
            target_wavelength = self.target_wavelength
        assert len(df1) == len(df2)

        calc = lambda x: m.log(x)
        first_night, second_night = list(map(calc, df1)), list(map(calc, df2))
        diff = np.subtract(second_night, first_night)
        diff = np.stack((self.w))

        if index is None:
            return diff
        else:
            self.new = np.stack((self.wl, first_night, second_night), axis=1)
            return self.new[self.target_wavelength, :]

    def calculate_radial_velocity(self, delta_lambda: float = None, lam: float = 6563):
        if delta_lambda is None:
            delta_lambda = self.new[self.target_wavelength, :]
        return const.c * (delta_lambda / lam)
