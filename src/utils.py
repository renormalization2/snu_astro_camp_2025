import os
import re

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.units as u

from src.constants import observatory


def isotime(time: str = None) -> str:
    """UTC time in ISO format"""
    time = time or Time(Time.now(), format="iso", scale="utc", location=observatory)
    if isinstance(time, str):  # e.g. "2025-08-03 12:00:00"
        fmt = "isot" if "T" in time else "iso"
        time = Time(time, format=fmt, scale="utc", location=observatory)
    return time.isot.split(".")[0]  # trim milliseconds


def ra_dec_to_l_b(ra: str | float, dec: float, time: str = None):
    """
    ra: str (hms) or float (deg)
    dec: float
    time: str or Time
    """
    time = time or isotime()

    angle_ra = Angle(ra, unit=u.hourangle if isinstance(ra, str) else u.deg)
    angle_dec = Angle(dec, unit=u.deg)

    coord_icrs = SkyCoord(ra=angle_ra, dec=angle_dec, frame="icrs", obstime=time, location=observatory)
    coord_gal = coord_icrs.galactic

    return coord_gal.l.value, coord_gal.b.value


def alt_az_to_l_b(alt, az, time: str = None):
    time = time or isotime()

    coord_aa = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame="altaz", obstime=time, location=observatory)
    coord_gal = coord_aa.galactic  # transform to Galactic
    return coord_gal.l.value, coord_gal.b.value


def l_b_to_alt_az(l, b, time: str = None):
    time = time or isotime()

    coord_gal = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic", obstime=time, location=observatory)
    coord_aa = coord_gal.transform_to("altaz")
    return coord_aa.alt.value, coord_aa.az.value


def unique_filename(filepath: str, always_add_counter: bool = False) -> str:
    """
    Given a desired filepath (can include directory and extension),
    returns a variant that does not collide with an existing file.
    If 'foo.txt' exists, will try 'foo_001.txt', 'foo_002.txt', etc.
    """
    directory, filename = os.path.split(filepath)
    stem, ext = os.path.splitext(filename)

    rx = re.compile(r"^(.+)_(\d+)$")
    match = rx.match(stem)

    # if counter exists
    if match:
        counter = int(match.group(2))
        counterless_stem = match.group(1)  # make stem counter-less
        candidate = f"{stem}_{counter:03d}{ext}"

    # if counter doesn't exist
    else:
        counter = 0 if always_add_counter else 1
        counterless_stem = stem
        # redefine filename with counter-added stem
        candidate = "".join([f"{counterless_stem}_{counter:03d}" if always_add_counter else stem, ext])

    # Loop until we find a non-existing filename
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{counterless_stem}_{counter:03d}{ext}"
        counter += 1

    return os.path.join(directory, candidate)


def calc_psd(sample, fs, fc=0, N_fft=1024, overlap=0, window_func=np.hamming):

    # Welch method - PSD averaging
    # overlap: between windows fraction
    # N_fft: length of each segments (higher value leads higher spectral resolution)

    win = window_func(N_fft)  # window function
    U = np.mean(win**2)  # power of window function

    step = int(N_fft * (1 - overlap))  # window step
    freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1 / (fs / 1e6))) + fc / 1e6  # frequency grid

    segments = []
    for i in range(0, len(sample) - N_fft + 1, step):
        seg = sample[i : i + N_fft]
        segments.append(seg * win)  # convolution in time domain
    segments = np.array(segments)

    # FFT for each segments
    F_seg = np.fft.fftshift(np.fft.fft(segments, axis=1), axes=1) / N_fft
    psd_seg = np.abs(F_seg) ** 2 / U
    psd = np.mean(psd_seg, axis=0)

    return freq, psd
