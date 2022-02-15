from pathlib import Path
import scipy.io.wavfile as wav
import scipy.signal as sig
from plot import plot_time_frequency
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Input
original_path = Path('original.wav')
resynthesized_path = Path('resynthesized.wav')

_, signal_original = wav.read(original_path)
fs, signal_resynthesized = wav.read(resynthesized_path)

# STFT
ts = 1 / fs  # in seconds
time_resolution = 0.01  # in seconds
t_fft = 0.1  # seconds
n_fft = int(fs * t_fft)  # in samples
padding_factor = 2
n_fft = n_fft * padding_factor  # in samples
frequency_precision = fs / n_fft  # in Hertz
win_length = int(t_fft * fs)
hop_length = int(fs * time_resolution)
window = 'hann'
eps = 1e-12
stft_parameters = {
    'fs': fs,
    'window': window,
    'nperseg': win_length,
    'noverlap': win_length - hop_length,
    'nfft': n_fft,
    'boundary': 'zeros',
}

start = time()
_, _, stft_original = sig.stft(signal_original, **stft_parameters)
omega, tau, stft_resynthesized = sig.stft(signal_resynthesized, **stft_parameters)
print('Time to STFT: %.3f' % (time() - start))

spectrogram_original = np.abs(stft_original)**2
spectrogram_original_db = 10 * np.log10(spectrogram_original + eps)

spectrogram_resynthesized = np.abs(stft_resynthesized)**2
spectrogram_resynthesized_db = 10 * np.log10(spectrogram_resynthesized + eps)

# Original
fig = plot_time_frequency(spectrogram_original_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                          time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
plt.tight_layout()
plt.savefig('figure_piano_original.eps', bbox_inches='tight', pad_inches=0, transparent=True)

# Resynthesized
fig = plot_time_frequency(spectrogram_resynthesized_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                          time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
plt.tight_layout()
plt.savefig('figure_piano_resynthesized.eps', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show()
