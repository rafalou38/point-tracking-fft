import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack
import numpy as np
from scipy.optimize import curve_fit


def expAmortie(t, a,tau,f0,phi,b):
    # a = 60
    # b = 4
    # f0 = 3.45
    # d = 1.9
    # e = 295
    return a * np.exp(-t / b) * np.cos(2 * np.pi * t * f0 + phi) + b


# Définis la fonction lorentzienne
def lorentz(x, amplitude, center, width):
    return amplitude / np.pi * (width / 2) / ((x - center) ** 2 + (width / 2) ** 2)
    # return (2/np.pi*width) / (1+((x-center)/(width/2))**2)


SKIP = 10

def viewData(file):
    # Data: t, x,y
    df = pd.read_csv(file+".csv")

    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.scatter(df["t"], df["y"], linewidth=1, s=4, marker="x")
    plt.plot(df["t"], df["y"], linewidth=1)

    plt.title("Y en fonction de t")

    plt.xlabel("t (s)")
    plt.ylabel("y (px)")

    # FT
    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(df["y"])
    xf = np.fft.fftfreq(len(yf), df["t"][1] - df["t"][0])[SKIP:-SKIP]
    yf = yf[SKIP:-SKIP]

    plt.scatter(np.abs(xf), np.abs(yf), linewidth=1, s=6, marker="x")
    plt.title("FFT de Y")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")

    # Modélisation
    try:
        lorentzOpts, pcov = curve_fit(lorentz, np.abs(xf), np.abs(yf))
        amplitude, center, width = lorentzOpts
        plt.annotate(
            f"f(x)=a/π*(w/2)/((x-c)^2+(w/x0)^2)\n\na = {amplitude:.2f}\nx0 = {center:.2f}\nw = {width:.2f}",
            xy=(0.75, 0.25),
            xycoords="subfigure fraction",
            fontsize=11,
        )
        aMax = np.max(np.abs(yf))

        print(amplitude, center, width, aMax)
        plt.plot(np.abs(xf), lorentz(np.abs(xf), *lorentzOpts), "r-", linewidth=1)

        # Display in graph (line)
        x1 = center - (width / 2) * np.sqrt((np.sqrt(2) - 1))
        x2 = center + (width / 2) * np.sqrt((np.sqrt(2) - 1))
        # x2 = x1 + center-x1

        print(x2 - x1)
        # plt.arrow(x1, aMax * 1/np.sqrt(2), 0, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        # plt.axline((x1, aMax * 1/np.sqrt(2)), (x2, aMax * 1/np.sqrt(2)), linewidth=1)
        plt.plot([x1, x2], [aMax * 1 / np.sqrt(2), aMax * 1 / np.sqrt(2)], linewidth=1)
        plt.plot([center, center], [aMax * 1 / np.sqrt(2), aMax], linewidth=1)

        plt.annotate(
            f"Fmax={center:.1f}Hz\n w(1/sqrt(2))={x2 - x1:.2f}Hz", xy=(x2 + 0.2, aMax * 1 / np.sqrt(2))
        )
    except Exception as e:
        print(e)

    plt.tight_layout()
    plt.savefig(file+".png", dpi=200)
    # plt.show()


if __name__ == "__main__":
    viewData("18-nov/18nov-libre plastique-3g0-2aimants.avi")
