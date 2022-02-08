from pathlib import Path
import scipy.io.wavfile as wav
import scipy.signal as sig
from generation_signal.signal_generation import pad_and_cut
from plot import plot_time_frequency, plot_time_frequency_2
import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy.ndimage.morphology as morpho
from synthesis import recover_vectors_bis, synthesize_from_arrays, synthesize_noise_mask
from utils import get_str_el, save_pickle

plot = False

# Input
signal_path = Path('clarinet.wav')
duration = 4.  # in seconds
init_rest = 1.  # in seconds
final_rest = 1.  # in seconds
fade_out = 0.1  # in seconds

fs, signal = wav.read(signal_path)
signal = pad_and_cut(signal, duration, fs, init_rest, final_rest, fade_out)

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
omega, tau, stft = sig.stft(signal, **stft_parameters)
print('Time to STFT: %.3f' % (time() - start))

spectrogram = np.abs(stft)**2
spectrogram_db = 10 * np.log10(spectrogram + eps)

# Morphology
start = time()

# Closing
closing_time_width = 0.05  # in seconds
closing_frequency_width = 50.  # in Hertz
closing_shape = (int(closing_frequency_width / frequency_precision), int(closing_time_width / time_resolution))
str_el_clo = np.zeros(closing_shape)

closing = morpho.grey_closing(spectrogram_db, structure=str_el_clo)

# Erosion
n_freq_bins_erosion = 5 * padding_factor

idx = np.arange(- n_freq_bins_erosion // 2 + 1, n_freq_bins_erosion // 2)
window_spectrum = get_str_el(window, fs, n_fft, factor=padding_factor, plot=False, eps=eps).astype(np.float32)
erosion_freq_shape = np.take(window_spectrum, idx)
str_el_ero = np.expand_dims(np.array(erosion_freq_shape), axis=1)

erosion = morpho.grey_erosion(closing, structure=str_el_ero)

# Top-hat
top_hat_width = 5  # in samples
top_hat_freq_width = top_hat_width * frequency_precision  # in Hertz
threshold = 5.  # in dB

str_el_clo = np.zeros((top_hat_width, 1))

top_hat = erosion - morpho.grey_opening(erosion, structure=str_el_clo)

output = np.copy(erosion)
output[top_hat < threshold] = -1000

# Opening
opening_frequency_width = 100  # in Hertz
opening_width = int(opening_frequency_width / frequency_precision)  # in samples

str_el_ope = np.zeros((opening_width, 1))

opening = morpho.grey_opening(closing, structure=str_el_ope)

print('Time to morphology: %.3f' % (time() - start))

# Recover parameters
threshold_amplitude = -120
threshold_duration = 0.05
neighbourhood_width = top_hat_width // 2 + 1

start = time()
spectrograms_for_synth = np.copy(output)
output_arrays = recover_vectors_bis(spectrograms_for_synth, tau, omega, time_resolution, neighbourhood_width,
                                    threshold_amplitude, threshold_duration)
print('Time to recover vectors: %.3f' % (time() - start))

# Write synthesis parameters
synthesis_parameters = {'harmonic': output_arrays, 'non-harmonic': opening}
synthesis_parameters_path = Path('synthesis_parameters.pickle')
save_pickle(synthesis_parameters_path, synthesis_parameters)

# Synthesis
start = time()
duration = duration + init_rest + final_rest  # in seconds
noise_normalization = 'max'
synthesized_harmonic, time_harmonic = synthesize_from_arrays(output_arrays, duration, fs)
synthesized_non_harmonic, time_noise, white_noise_stft, filtered_noise_stft = synthesize_noise_mask(opening, duration,
                                                                                                    noise_normalization,
                                                                                                    **stft_parameters)
white_noise_db = 10 * np.log10(np.abs(white_noise_stft)**2 + eps)
filtered_noise_db = 10 * np.log10(np.abs(filtered_noise_stft)**2 + eps)
print('Time to synthesize: %.3f' % (time() - start))

# Write audio
original_path = Path('original.wav')
harmonic_path = Path('harmonic.wav')
non_harmonic_path = Path('non-harmonic.wav')
resynthesized_path = Path('resynthesized.wav')

wav.write(original_path, fs, signal.astype(np.float32))
wav.write(harmonic_path, fs, synthesized_harmonic.astype(np.float32))
wav.write(non_harmonic_path, fs, synthesized_non_harmonic.astype(np.float32))
wav.write(resynthesized_path, fs, (synthesized_harmonic + synthesized_non_harmonic).astype(np.float32))

# Plot
if plot:
    # Input
    fig = plot_time_frequency(spectrogram_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_input.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Closing
    fig = plot_time_frequency_2(spectrogram_db, closing, tau, omega, v_min=-120, v_max=0, resolution='s',
                                time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0.9 / time_resolution, 2. / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 500. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_closing.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Erosion
    fig = plot_time_frequency_2(closing, erosion, tau, omega, v_min=-120, v_max=0, resolution='s',
                                time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([2.2 / time_resolution, 2.4 / time_resolution])
    fig.axes[0].set_ylim([170. * (t_fft * padding_factor), 270. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_erosion.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Top-hat
    fig = plot_time_frequency(top_hat, tau, omega, v_min=0, v_max=20, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_top-hat.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Opening
    fig = plot_time_frequency(opening, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_opening.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # White noise
    fig = plot_time_frequency(white_noise_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_white_noise.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Filtered noise
    fig = plot_time_frequency(filtered_noise_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=(600, 300), show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    plt.savefig('figure_filtered_noise.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Show
    plt.show()
