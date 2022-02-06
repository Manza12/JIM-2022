import numpy as np
from scipy.signal.windows import get_window


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
