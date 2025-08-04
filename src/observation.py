import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from rtlsdr import RtlSdr

from src.constants import NFFT, DATA_DIR, DEMO_DATA_DIR
from src.utils import unique_filename, alt_az_to_l_b, ra_dec_to_l_b, isotime

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
        filename = f"{time}_{x}_{y}_{suffix}.csv" if suffix else f"{time}_{x}_{y}.csv"

    save_dir = DATA_DIR if not demo else DEMO_DATA_DIR
    tbl = Table()
    tbl["frequency"] = freq
    tbl["power"] = power
    tbl.write(unique_filename(save_dir / filename), format="csv", delimiter="\t")
    # np.savetxt(unique_filename(save_dir / filename), np.stack([freq, power]), delimiter=",")


def load_data(filename=None, time=None, x=None, y=None, demo=False):
    filename = filename or f"{time}_{x}_{y}.csv"
    if os.path.dirname(filename) == "":
        filename = DATA_DIR / filename if not demo else DEMO_DATA_DIR / filename

    if os.path.splitext(filename)[1] == ".npy":
        return np.load(filename)
    return np.loadtxt(filename, delimiter=",")


class Exposure:
    """Exposures whose pointing and obstime can be considered the same"""

    def __init__(self, n_obs=10, alt=None, az=None, ra=None, dec=None, l=None, b=None, time=None, type=None):
        self.n_obs = n_obs
        self.time = time or isotime()
        self.exposure_type = type  # e.g., sky, ground

        self.alt = alt
        self.az = az
        self.ra = ra
        self.dec = dec
        self.l = l
        self.b = b
        if self.alt and self.az:
            self.l, self.b = alt_az_to_l_b(self.alt, self.az, time)
        elif self.ra and self.dec:
            self.l, self.b = ra_dec_to_l_b(self.ra, self.dec, time)
        else:
            print("[Warning] No RA/Dec or Alt/Az pair provided")
            self.l = "unknown"
            self.b = "coordinates"

        # canvas for spectrum plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

    def run(self):
        plt.ion()  # plt interactive mode on
        for i in range(self.n_obs):
            samples = self._expose()
            power, freq = self._get_spectrum(samples)
            save_spectrum(freq, power, time=self.time, x=self.l, y=self.b, suffix=self.exposure_type)
        plt.ioff()

    def _expose(self, sample_rate=3e6, center_freq=1.4204e9, gain=50, n_samples=256 * NFFT, save_raw=False):
        """unload raw time-series data from memory immediately, and only keep the power spectrum"""
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.n_samples = n_samples
        self.n_fft = NFFT
        self.freq = None
        self.powers = np.empty((self.n_obs, self.n_fft))

        samples = expose_sdr(
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            gain=self.gain,
            n_samples=self.n_samples,
        )
        if save_raw:
            fname = unique_filename(DATA_DIR / f"{self.time}_{self.l}_{self.b}_raw.npy", always_add_counter=True)
            np.save(fname, samples)

        return samples

    def _get_spectrum(self, samples):
        fig, ax = plt.subplots(figsize=(8, 6))
        # ax.psd includes Hann windowing and gives a less noisy spectrum than np.fft
        power, freq = ax.psd(
            samples,
            NFFT=self.n_fft,
            Fs=self.sample_rate / 1e6,
            Fc=self.center_freq / 1e6,
        )  # 1e6 for MHz
        ax.set_xlim(np.min(freq), np.max(freq))
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB/MHz)")
        ax.minorticks_on()

        # Draw & pause briefly so GUI can update (necessary in interactive mode)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if not self.freq:
            self.freq = freq
        assert self.freq == freq  # ensure the same frequency grid is used for all exposures
        self.powers.append(power)
        return power, freq

    @property
    def power(self):
        if self.powers is None:
            raise ValueError("No power spectrum data available")
        return np.mean(self.powers, axis=0)

    def load(self, filenames: list = None, time=None, x=None, y=None, demo=False):
        if filenames and isinstance(filenames, str):
            filenames = [filenames]

        if os.path.dirname(filename) == "":
            filename = DATA_DIR / filename if not demo else DEMO_DATA_DIR / filename

        template = f"*_{self.l}_{self.b}_{self.exposure_type}*.csv"
        flist = glob(template)
        for f in flist:
            data = load_data(f)
            self.powers.append(data["power"])

        return data["frequency"], self.powers
