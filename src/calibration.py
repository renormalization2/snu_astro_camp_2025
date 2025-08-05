import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.constants import DATA_DIR


def calibrate(sky, ground, plot=True):
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

    np.save(DATA_DIR / f"TA_{sky.l}_{sky.b}.npy", T_src)

    return freq, T_src


def gaussian_fit(freq, T_src, plot=True):
    pass
