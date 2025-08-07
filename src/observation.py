from fileinput import filename
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from rtlsdr import RtlSdr

from src.constants import NFFT, DATA_DIR, DEMO_DATA_DIR
from src.utils import unique_filename, alt_az_to_l_b, ra_dec_to_l_b, isotime, clean_isot, restore_isot

# [Warm Up] this builds the transform graph and prevent overhead in the subsequent code
_ = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic").icrs


def expose_sdr(sample_rate=3e6, center_freq=1.4204e9, gain=50, n_samples=256 * NFFT):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate  # equals bandwidth (Hz) in complex (IQ) sampling (Nyquist-Shannon)
    sdr.center_freq = center_freq  # center frequency (Hz)
    sdr.freq_correction = 1  # no need to correct
    sdr.gain = gain  # find the highest value before saturation
    samples = sdr.read_samples(n_samples)
    sdr.close()
    return samples


def save_spectrum(freq, power, time=None, x=None, y=None, suffix=None, demo=False):
    time = time or isotime()
    if x is None and y is None:
        filename = f"{time}_{suffix}.csv" if suffix else f"{time}.csv"
    else:
        filename = f"{time}_{x:.0f}_{y:.0f}_{suffix}.csv" if suffix else f"{time}_{x:.0f}_{y:.0f}.csv"

    filename = clean_isot(filename)

    save_dir = DATA_DIR if not demo else DEMO_DATA_DIR
    tbl = Table()
    tbl["frequency"] = freq
    tbl["power"] = power
    tbl.write(unique_filename(save_dir / filename, always_add_counter=True), format="csv", delimiter="\t")
    # np.savetxt(unique_filename(save_dir / filename), np.stack([freq, power]), delimiter=",")


def load_data(filename=None, time=None, l=None, b=None, demo=False):
    filename = filename or f"{time}_{l:.0f}_{b:.0f}.csv"
    if os.path.dirname(filename) == "":
        filename = DATA_DIR / filename if not demo else DEMO_DATA_DIR / filename

    if os.path.splitext(filename)[1] == ".npy":
        return np.load(filename)
    # return np.loadtxt(filename, delimiter=",")
    return Table.read(filename, format="csv", delimiter="\t")


class Exposure:
    """Exposures whose pointing and obstime can be considered the same"""

    def __init__(
        self, n_obs=10, alt=None, az=None, ra=None, dec=None, l=None, b=None, time=None, type=None, n_fft=NFFT
    ):
        self.n_obs = n_obs
        self.time = time or isotime()
        self.exposure_type = type  # e.g., sky, ground
        self.n_fft = n_fft

        self.alt = alt
        self.az = az
        self.ra = ra
        self.dec = dec
        if self.alt and self.az:
            self.l, self.b = alt_az_to_l_b(self.alt, self.az, self.time)
        elif self.ra and self.dec:
            self.l, self.b = ra_dec_to_l_b(self.ra, self.dec, self.time)
        else:
            print("[Warning] No RA/Dec or Alt/Az pair provided")
            self.l = l
            self.b = b

        if self.l and self.b:
            print(f"Data will be saved at [{self.time}] with l={self.l:.0f}, b={self.b:.0f}")
        else:
            print(f"Data will be saved at [{self.time}] with unknown coordinates")

        self.freq = None
        self.powers = np.empty((self.n_obs, self.n_fft))

    def __repr__(self):
        return f"Exposure object with n_obs={self.n_obs}, l={self.l}, b={self.b}, time={self.time}"

    def run(self, demo=False, gain=None):
        # canvas for spectrum plot
        if demo:
            flist = glob(str(DEMO_DATA_DIR / "2025-08-04" / "*_31_11_sky_???.csv"))[:10]
            for i, f in enumerate(flist):
                data = load_data(f)
                power, freq = np.array(data["power"]), np.array(data["frequency"])
                self.powers[i] = power
            self.freq = freq
        else:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            # plt.ion()  # plt interactive mode on
            for i in range(self.n_obs):
                samples = self._expose(
                    sample_rate=3e6,
                    center_freq=1.4204e9,
                    gain=gain or 50,
                    n_samples=256 * self.n_fft,
                    save_raw=False,
                )
                freq, power = self._get_spectrum(i, samples)
                save_spectrum(freq, power, time=self.time, x=self.l, y=self.b, suffix=self.exposure_type)
            # plt.ioff()

        # for power in self.powers:
        #     self.ax.plot(freq, power, c=f"C{i}", alpha=0.5)
        # self.ax.set_xlim(np.min(freq), np.max(freq))
        # self.ax.set_xlabel("Frequency (MHz)")
        # self.ax.set_ylabel("Power (dB/MHz)")
        # self.ax.minorticks_on()
        # print(np.min(self.freq))
        # self.ax.plot(self.freq, self.power, c="k", label="Mean Spectrum")  # MHz
        # self.ax.legend()
        print("Exposure finished")

    def _expose(self, sample_rate=3e6, center_freq=1.4204e9, gain=50, n_samples=256 * NFFT, save_raw=False, **kwargs):
        """unload raw time-series data from memory immediately, and only keep the power spectrum"""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.n_samples = n_samples
        print(f"gain: {self.gain}")
        # self.powers = np.empty((self.n_obs, self.n_fft))  # redefine with new n_fft

        samples = expose_sdr(
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            gain=self.gain,
            n_samples=self.n_samples,
        )
        if save_raw:
            # fname = unique_filename(DATA_DIR / f"{self.time}_{self.l}_{self.b}_raw.npy", always_add_counter=True)
            time = clean_isot(self.time)
            fname = unique_filename(DATA_DIR / f"{time}_{self.l}_{self.b}_raw.npy", always_add_counter=True)
            np.save(fname, samples)

        return samples

    def _get_spectrum(self, i, samples):
        # self.ax.cla()  # clear previous axis
        fig, ax = self.fig, self.ax
        # ax.psd includes Hann windowing and gives a less noisy spectrum than np.fft
        # _fig, _ax = plt.subplots()
        # power, freq = _ax.psd(
        power, freq = ax.psd(
            # power, freq = psd(
            samples,
            NFFT=self.n_fft,
            Fs=self.sample_rate / 1e6,
            Fc=self.center_freq / 1e6,
            # noverlap=0,
            # scale_by_freq=False,
        )  # 1e6 for MHz
        # ax.plot(freq, power, c=f"C{i}", alpha=0.5)
        # plt.close(_fig)  # kill the figure before showing
        # ax.plot(freq, power, c=f"C{i}", alpha=0.5)

        ax.set_xlim(np.min(freq), np.max(freq))
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB/MHz)")
        ax.minorticks_on()

        # Draw & pause briefly so GUI can update (necessary in interactive mode)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.freq is None:
            self.freq = freq
        assert np.array_equal(self.freq, freq)  # ensure the same frequency grid is used for all exposures
        self.powers[i] = power
        return freq, power

    @property
    def power(self):
        if self.powers is None:
            raise ValueError("No power spectrum data available")
        return np.mean(self.powers, axis=0)

    def load(self, filenames: list = None, time=None, l=None, b=None, type=None, demo=False):
        if filenames and isinstance(filenames, str):
            filenames = [filenames]

        l = l or self.l
        b = b or self.b
        exposure_type = type or self.exposure_type

        time_str = clean_isot(time) if time is not None else "*"
        l_str = f"_{l:.0f}" if l is not None else "*"
        b_str = f"_{b:.0f}" if b is not None else "*"
        exp_str = exposure_type if exposure_type is not None else "*"
        template = f"{time_str}{l_str}{b_str}_{exp_str}*.csv"
        template = str((DEMO_DATA_DIR if demo else DATA_DIR) / template)
        flist = glob(template)
        print(f"loading {len(flist)} files")

        self.n_obs = len(flist)
        self.powers = np.empty((self.n_obs, self.n_fft))

        for i, f in enumerate(flist):
            data = load_data(f)
            self.powers[i] = np.array(data["power"])

        self.freq = np.array(data["frequency"])

        if not hasattr(self, "time") or (hasattr(self, "time") and self.time is None):
            from pathlib import Path

            time_str = restore_isot(Path(f).stem.split("_")[0])
            self.time = isotime(time_str)

        return self.freq, self.powers

    @classmethod
    def from_file(cls, time=None, l=None, b=None, type=None, n_fft=NFFT, demo=False):
        self = cls.__new__(cls)  # (l=l, b=b, time=time, type=type, demo=demo)
        self.l = l
        self.b = b
        self.n_fft = n_fft
        self.exposure_type = type
        self.load(l=l, b=b, type=self.exposure_type, demo=demo)
        return self

    def plot_spectrum(self, ax=None, show=True):
        from matplotlib.collections import LineCollection
        from matplotlib.legend_handler import HandlerLineCollection

        # color palette
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, len(self.powers)))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        for power, col in zip(self.powers, colors):
            ax.plot(self.freq, 10 * np.log10(power), alpha=0.5, color=col)
        segs = [([[0, 0], [1, 0]]) for _ in colors]  # one horizontal segment per color
        indiv_spec_label_proxy = LineCollection(segs, colors=colors, linewidths=7)
        indiv_spec_label_proxy.set_label("Individual Spectra")

        (mean_spec_label,) = ax.plot(self.freq, 10 * np.log10(self.power), c="k", label="Mean Spectrum")

        # multi-color legend
        ax.legend(
            handles=[indiv_spec_label_proxy, mean_spec_label],
            handler_map={LineCollection: HandlerLineCollection(numpoints=len(colors))},
            loc="best",
            frameon=False,
        )
        ax.set_xlim(np.min(self.freq), np.max(self.freq))
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB/MHz)")
        if self.exposure_type:
            ax.set_title(f"{self.exposure_type.title()}")
        ax.minorticks_on()
        ax.grid(which="both")
        if show:
            plt.show()
        return fig, ax


def cosmetics(ax):
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dB/MHz)")
    # if self.exposure_type:
    #     ax.set_title(f"{self.exposure_type.title()}")
    ax.minorticks_on()
    ax.grid(which="both")
    plt.show()


#     def plot_spectrum(self):
#         cmap = plt.cm.viridis
#         colors = cmap(np.linspace(0, 1, len(self.powers)))

#         fig, ax = plt.subplots(figsize=(8, 6))
#         for power, col in zip(self.powers, colors):
#             ax.plot(self.freq, 10 * np.log10(power), color=col, alpha=0.5)

#         ax.plot(self.freq, 10 * np.log10(self.power), "k", lw=2, label="Mean Spectrum")

#         # add dummy Rectangle that carries the colormap
#         grad_handle = Rectangle((0, 0), 1, 1, facecolor="none")
#         grad_handle.cmap = cmap
#         ax.legend(
#             [
#                 grad_handle,
#             ],
#             ["Individual Spectra"],
#             handler_map={Rectangle: HandlerGradient()},
#             frameon=False,
#         )

#         ax.set_xlim(self.freq.min(), self.freq.max())
#         ax.set_xlabel("Frequency (MHz)")
#         ax.set_ylabel("Power (dB/MHz)")
#         ax.minorticks_on()
#         ax.grid(which="both", alpha=0.3)
#         return fig, ax


# from matplotlib.patches import Rectangle
# from matplotlib.legend_handler import HandlerPatch


# class HandlerGradient(HandlerPatch):
#     def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
#         # orig_handle is a dummy Rectangle
#         grad = np.linspace(0, 1, 256).reshape(1, -1)
#         ax = legend.axes
#         im = ax.imshow(
#             grad,
#             cmap=orig_handle.cmap,
#             extent=(xdescent, xdescent + width, ydescent, ydescent + height),
#             transform=trans,
#             aspect="auto",
#         )
#         return [im]
