from utils import plot_window
import matplotlib.pyplot as plt

window = 'hann'
fs = 44100
time_resolution = 0.01  # in seconds
t_fft = 0.1  # seconds
n_fft = int(fs * t_fft)  # in samples
padding_factor = 2
n_freq_bins_erosion = 5 * padding_factor

if __name__ == '__main__':
    plot_window(window, fs, n_fft, n_freq_bins_erosion, factor=padding_factor)
    plt.show()
