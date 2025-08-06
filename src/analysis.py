import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18
plt.rcParams["legend.frameon"] = False


def single_gauss(x, mu1, sigma1, A1):
    return A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)


def double_gauss(x, mu1, sigma1, A1, mu2, sigma2, A2):
    g1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    g2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    return g1 + g2


def triple_gauss(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
    g1 = A1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    g2 = A2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    g3 = A3 * np.exp(-0.5 * ((x - mu3) / sigma3) ** 2)
    return g1 + g2 + g3


def gaussian_fit(V_r, T_src, p0, obj=None, plot=True, return_plot=True):
    if len(p0) == 1:
        func = single_gauss
    elif len(p0) == 2:
        func = double_gauss
    elif len(p0) == 3:
        func = triple_gauss
    else:
        raise ValueError("Invalid number of parameters")

    p0 = np.array(p0).flatten()

    # Fit
    popt, pcov = curve_fit(func, V_r, T_src, p0=p0)

    # Prepare high-res x
    x_fit = np.linspace(V_r.min(), V_r.max(), 2000)

    # total model
    y_tot = func(x_fit, *popt)

    # Extract fitted parameters and uncertainties in mu, sigma, A order
    perr = np.sqrt(np.diag(pcov))
    for i, (mu, sig, amp, mu_err, sig_err, amp_err) in enumerate(
        zip(popt[0::3], popt[1::3], popt[2::3], perr[0::3], perr[1::3], perr[2::3]), start=1
    ):
        print(
            f"Component {i}: μ = {mu:.2f} ± {mu_err:.2f}, "
            f"σ = {sig:.2f} ± {sig_err:.2f}, "
            f"A = {amp:.2f} ± {amp_err:.2f}"
        )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        # label = rf"$T_{{src}}$ ($\ell = {obj.l}^\circ$, $b = {obj.b}^\circ$)" if obj else rf"$T_{{src}}$"
        label = rf"$T_{{src}}$"
        ax.plot(V_r, T_src, lw=1, label=label)
        ax.plot(x_fit, y_tot, "r--", label="Total fit", lw=2)

        # Components
        for i in range(len(popt) // 3):
            mu, sigma, A = popt[i * 3], popt[i * 3 + 1], popt[i * 3 + 2]
            y = A * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)
            ax.plot(x_fit, y, label=f"Component {i+1}")

        ax.axvline(0, color="k", ls=":", label="v=0 km/s")
        if obj:
            ax.set_title(rf"$\ell = {obj.l}^\circ$, $b = {obj.b}^\circ$")
        ax.set_xlabel(r"$\rm V_r\ [km\ s^{-1}]$")
        ax.set_ylabel(r"$T_A$ (K)")
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        if return_plot:
            return fig, ax

    return popt, perr


from astropy.table import Table
from src.constants import DEMO_DATA_DIR

tbl = Table.read(DEMO_DATA_DIR / "rotation_curve.csv")


def plot_archive_rotation_curve():
    from scipy.interpolate import CubicSpline

    plt.errorbar(
        tbl["r"],
        tbl["V"],
        yerr=tbl["dV"],
        color="k",
        label="Bhattacharjee et al.(2014)",
        ls="None",
        capsize=3,
        marker="o",
        ms=3,
        lw=0.5,
    )

    func = CubicSpline(tbl["r"], tbl["V"], bc_type="natural")
    r0 = 8.5
    rs = np.arange(0, 15, 0.1)
    plt.plot(rs, func(rs), alpha=0.5)  # , label="spline curve")

    plt.axvline(r0, color="k", ls="dotted", label="Solar Position")
    plt.minorticks_on()
    plt.ylabel(r"Rotation Velocty $V_{LSR}(r)\;[\rm kms^{-1}]$")
    plt.xlabel(r"Radius $r\;[\rm kpc]$")
    plt.xlim(-0.5, 15)
    plt.ylim(0, 300)
    plt.legend(loc="lower right")
