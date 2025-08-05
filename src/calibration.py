import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from src.constants import DATA_DIR, DEMO_DATA_DIR, observatory
from src.observation import Exposure


def calibrate():
    pass


def power_to_TA(sky, ground, plot=True):
    # T_sky = 20  # Kelvin
    T_amb = 300  # 273.15 K + 30 K

    P_amb = ground.power
    P_src = sky.power
    freq = sky.freq

    Y = np.median(P_amb / P_src)
    P_sky = P_amb / Y
    # T_sys = (T_amb - Y * T_sky) / (Y - 1)

    if plot:
        plt.plot(freq, 10 * np.log10(P_src), label="Source")
        plt.plot(freq, 10 * np.log10(P_amb), label="Ambient")
        plt.plot(freq, 10 * np.log10(P_sky), label="Sky (scaled Ambient)")
        plt.ylabel("Power (dB/MHz)")
        plt.xlabel("Frequency (MHz)")
        plt.title(f"Y = {Y:.2f}")
        plt.legend()
        plt.show()

    T_src = (P_src - P_sky) / (P_amb - P_sky) * T_amb

    if plot:
        plt.plot(freq, T_src, label=rf"$T_{{src}}$ ($\ell = {sky.l}$, $b = {sky.b}$)")
        plt.ylabel(r"$T_A$ (K)")
        plt.xlabel("Frequency (MHz)")
        plt.legend()
        plt.show()

    np.save(DATA_DIR / "freq.npy", freq)
    np.save(DATA_DIR / f"TA_{sky.l}_{sky.b}.npy", T_src)

    return freq, T_src


def freq_to_velocity(freq):
    f0 = 1420.40575  # rest frame frequency of H1
    c = 2.9979e5  # lightspeed in [km/s]
    return -c * (freq - f0) / freq


def get_v_corr(sky):
    obstime = sky.time
    l = sky.l
    b = sky.b

    obj = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic", obstime=obstime, location=observatory)

    # kinematic LSR correction (astropy version dependent)
    # vcorr_lsr = obj.radial_velocity_correction(kind="lsrk").to(u.km / u.s).value
    helio_corr = obj.radial_velocity_correction("heliocentric").to(u.km / u.second).value

    # Peculiar velocity of sun
    # v_sun = [10, 5, 7] * u.km / u.s
    U, V, W = 10, 5, 7
    peculiar_corr = U * np.sin(l) + V * np.cos(l)
    v_corr = helio_corr + peculiar_corr
    return v_corr


def load_TA(l, b, type="sky", demo=False, velocity=False):
    # freq = Exposure.from_file(l=l, b=b, type=type, demo=demo).freq
    data_dir = DATA_DIR  # if not demo else DEMO_DATA_DIR   # always datadir for .npy
    TA = np.load(data_dir / f"TA_{l}_{b}.npy")
    if not velocity:
        freq = np.load(data_dir / "freq.npy")
        return freq, TA
    else:
        velocity = np.load(data_dir / f"Vr_{l}_{b}.npy")
        return velocity, TA
