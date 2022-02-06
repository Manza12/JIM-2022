import numpy as np
from scipy.signal.windows import get_window
import os
import os.path as path
from pathlib import Path
import pickle


def get_str_el(window, fs, n_fft, factor=1, plot=True, db=True, eps=1e-10):
    window_time = get_window_dispatch(window, n_fft)
    window_norm = window_time / np.sum(window_time)
    window_spectrum = np.abs(np.fft.fft(window_norm, factor * n_fft))

    if db:
        window_spectrum = 10 * np.log10(window_spectrum**2 + eps)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.fft.fftshift(np.fft.fftfreq(n_fft * factor, 1 / fs)), np.fft.fftshift(window_spectrum))
        plt.scatter(np.fft.fftshift(np.fft.fftfreq(n_fft * factor, 1 / fs)), np.fft.fftshift(window_spectrum),
                    color='r')

    return window_spectrum


def plot_window(window, fs, n_fft, n_freq_bins_erosion, factor=1, smooth_factor=10, eps=1e-15):
    window_time = get_window_dispatch(window, n_fft)
    window_norm = window_time / np.sum(window_time)
    window_spectrum = np.abs(np.fft.fft(window_norm, factor * n_fft))
    window_spectrum = 10 * np.log10(window_spectrum**2 + eps)

    window_spectrum_smooth = np.abs(np.fft.fft(window_norm, smooth_factor * factor * n_fft))
    window_spectrum_smooth = 10 * np.log10(window_spectrum_smooth ** 2 + eps)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4., 3.))
    plt.plot(np.fft.fftshift(np.fft.fftfreq(n_fft * factor * smooth_factor, 1 / fs)),
             np.fft.fftshift(window_spectrum_smooth))
    plt.scatter(np.fft.fftshift(np.fft.fftfreq(n_fft * factor, 1 / fs)[- n_freq_bins_erosion // 2 + 1:]),
                np.fft.fftshift(window_spectrum[- n_freq_bins_erosion // 2 + 1:]), color='r')
    plt.scatter(np.fft.fftshift(np.fft.fftfreq(n_fft * factor, 1 / fs)[: n_freq_bins_erosion // 2]),
                np.fft.fftshift(window_spectrum[: n_freq_bins_erosion // 2]), color='r')

    plt.xlim([-50, 50])
    plt.xlabel('Fréquence (Hz)')

    plt.ylim([-120, 5])
    plt.ylabel('Puissance (dB)')

    plt.tight_layout()

    plt.savefig('figure_window.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    return window_spectrum


def get_window_dispatch(window, n, fft_bins=True):
    if isinstance(window, str):
        return get_window(window, n, fftbins=fft_bins)
    elif isinstance(window, tuple):
        if window[0] == 'gaussian':
            assert window[1] >= 0
            sigma = np.floor(- n / 2 / np.sqrt(- 2 * np.log(10**(- window[1] / 20))))
            return get_window(('gaussian', sigma), n, fftbins=fft_bins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


def create_if_not_exists(folder: Path):
    if not path.exists(folder):
        os.makedirs(folder)


def save_pickle(file_path, data):
    with open(file_path, "wb") as fp:
        pickle.dump(data, fp)
