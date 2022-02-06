import scipy.io.wavfile as wav
import scipy.signal as sig
from pathlib import Path
from plot import plot_time_frequency
import numpy as np
import matplotlib.pyplot as plt

# Read files
folder = Path('..') / Path('output') / Path('synthesized')
file_name = 'signal_inharmonic.wav'
file_path = folder / file_name

fs, signal_harmonic = wav.read(file_path)

# STFT
t_fft = 0.1  # in seconds
t_hop = 0.01  # in seconds
n_per_seg = int(t_fft * fs)
n_overlap = int((t_fft-t_hop) * fs)
eps = 1e-12

omega, tau, stft = sig.stft(signal_harmonic, fs=fs, window='hann', nperseg=n_per_seg, noverlap=n_overlap)
spectrogram = np.abs(stft)**2
spectrogram_db = 10 * np.log10(spectrogram + eps)

fig = plot_time_frequency(spectrogram_db, tau, omega, v_min=-120, v_max=0, resolution='s', time_label='Temps (s)',
                          freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)

fig.axes[0].set_xlim([0.8 / t_hop, 2.7 / t_hop])
fig.axes[0].set_ylim([0. * t_fft, 1000. * t_fft])
plt.tight_layout()

plt.savefig('figure_inharmonic.eps', bbox_inches='tight', pad_inches=0, transparent=True)

plt.show()
