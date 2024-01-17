import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack
import numpy as np
from scipy.optimize import curve_fit


# >> FONCTIONS DE MODÉLISATION
def expAmortie(t, a, tau, f0, phi, b):
    return a * np.exp(-t / tau) * np.cos(2 * np.pi * t * f0 + phi) + b


def lorentz(x, amplitude, f0, df):
    return amplitude / np.sqrt(1 + ((x - f0) / (df / 2)) ** 2)


def viewData(file, withModel=True, withFFT=True):
    # Data: t, x,y
    df = pd.read_csv(file + ".csv", nrows=600)


    if withFFT:
        plt.figure(figsize=(12.5, 8))
        plt.subplot(2, 1, 1)
    else:
        plt.figure(figsize=(12.5, 5))
    plt.scatter(df["t"], df["y"], linewidth=1, s=4, marker="x")

    if withModel:
        try:
            print("Modélisation exponentielle amortie")
            expOpts, _ = curve_fit(
                expAmortie,
                np.array(np.abs(df["t"])),
                np.array(np.abs(df["y"])),
                np.array([
                    -16.5,  # a
                    2.3,  # tau
                    7.8,  # f0
                    -0.1,  # phi
                    97,  # b
                ]),
            )
            plt.plot(np.array(df["t"]), expAmortie(np.array(df["t"]), *expOpts), "r-", linewidth=1)
        except Exception as e:
            print(e)
            print("Erreur modélisation exponentielle amortie")
            pass

    plt.title("Y en fonction de t")

    plt.xlabel("t (s)")
    plt.ylabel("y (px)")

    if withFFT:
        # FT
        FT_LO = 10
        FT_HI = 50

        # Modélisation
        try:
            plt.subplot(2, 1, 2)
            yf = scipy.fftpack.fft(df["y"])
            xf = np.fft.fftfreq(len(yf), df["t"][1] - df["t"][0])[FT_LO:FT_HI]
            yf = yf[FT_LO:FT_HI]
            np.savetxt(file + ".raw-fft.csv", np.column_stack((np.abs(xf), np.abs(yf))), delimiter=',', header='x,y', comments='')
            
            plt.scatter(np.abs(xf), np.abs(yf), linewidth=1, s=6, marker="x")
            plt.title("Transformée de Fourier de y")
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            # Résolution de la modélisation
            if withModel:
                xm = np.linspace(3, 16, 1000)
                lorentzOpts, _ = curve_fit(lorentz, np.abs(xf), np.abs(yf))
                ym = lorentz(np.abs(xm), *lorentzOpts)
                np.savetxt(file + ".fft.csv", np.column_stack((xm,ym)), delimiter=',', header='x,y', comments='')

                amplitude, f0, df = lorentzOpts
                plt.annotate(
                    f"f(f)=a/sqrt(1+((f-f0)/(Δf/2))^2)\n\na = {amplitude:.2f}\nf₀ = {f0:.2f}\nΔf = {df:.2f}",
                    xy=(0.75, 0.25),
                    xycoords="subfigure fraction",
                    fontsize=11,
                )
                aMax = np.max(np.abs(ym))

                print(amplitude, f0, df, aMax)
                plt.plot(np.abs(xm), ym, "r-", linewidth=1)

                # Display in graph (line)
                x1 = f0 - df / 2
                x2 = f0 + df / 2

                print("df", x2 - x1)
                plt.plot(
                    [x1, x2],
                    [aMax * 1 / np.sqrt(2), aMax * 1 / np.sqrt(2)],
                    linewidth=1,
                )
                plt.plot([f0, f0], [aMax * 1 / np.sqrt(2), aMax], linewidth=1)

                plt.annotate(
                    f"f₀={f0:.1f}Hz\n Δf={df:.2f}Hz",
                    xy=(x2 + 0.2, aMax * 1 / np.sqrt(2)),
                )
        except Exception as e:
            print(e)

    plt.tight_layout()
    plt.savefig(file + ".png", dpi=200)
    plt.show()


if __name__ == "__main__":
    viewData("18-nov/Sans nom 1.avi")
