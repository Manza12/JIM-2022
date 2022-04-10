from pathlib import Path
import scipy.io.wavfile as wav
import scipy.signal as sig
from generation_signal.signal_generation import pad_and_cut
from plot import plot_time_frequency, plot_time_frequency_2, plot_time_frequency_top_hat
import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy.ndimage.morphology as morpho
from synthesis import synthesize_from_arrays, synthesize_noise_mask, ridge_following
from utils import get_str_el, save_pickle
from skeleton import skeleton

plot = True
show = True
fig_size = (640, 360)

# Input
signal_path = Path('violin_56.wav')
duration = 4.  # in seconds
init_rest = 1.  # in seconds
final_rest = 1.  # in seconds
fade_out = 0.1  # in seconds
duration_synth = duration + init_rest + final_rest  # in seconds

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
print('Time to STFT of input: %.3f' % (time() - start))

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

# # Erosion
# n_freq_bins_erosion = 5 * padding_factor
#
# idx = np.arange(- n_freq_bins_erosion // 2 + 1, n_freq_bins_erosion // 2)
# window_spectrum = get_str_el(window, fs, n_fft, factor=padding_factor, plot=False, eps=eps).astype(np.float32)
# erosion_freq_shape = np.take(window_spectrum, idx)
# str_el_ero = np.expand_dims(np.array(erosion_freq_shape), axis=1)
#
# erosion = morpho.grey_erosion(closing, structure=str_el_ero)

# Top-hat
top_hat_width = 5  # in samples
top_hat_freq_width = top_hat_width * frequency_precision  # in Hertz
threshold = 2.  # in dB

str_el_top_hat = np.zeros((top_hat_width, 1))

top_hat = closing - morpho.grey_opening(closing, structure=str_el_top_hat)

# Binarize top-hat
top_hat_binary = np.zeros_like(closing, dtype=np.int8)
top_hat_binary[top_hat > threshold] = 1

# Skeletonize the top-hat
str_el_skull = np.ones((2, 1), dtype=bool)

top_hat_skeleteon = skeleton(top_hat_binary, str_el_skull)


# # Close the top-hat
# clo_th_time_width = 0.05
# str_el_clo_th = np.zeros((1, int(clo_th_time_width / time_resolution)))
# top_hat_closed = morpho.grey_closing(top_hat, structure=str_el_clo_th)

# Create detector food image
detector_food = np.copy(top_hat_skeleteon)

# Opening
opening_frequency_width = 100  # in Hertz
opening_width = int(opening_frequency_width / frequency_precision)  # in samples

str_el_ope = np.zeros((opening_width, 1))

opening = morpho.grey_opening(closing, structure=str_el_ope)

print('Time to morphology: %.3f' % (time() - start))

# Recover parameters
threshold_duration = 0.1
linking_time = 0.1
neighbourhood_width = top_hat_width*3

start = time()
output_arrays = ridge_following(detector_food, tau, omega, neighbourhood_width,
                                threshold_duration, linking_time, time_resolution)
print('Time to recover vectors: %.3f' % (time() - start))

# Write synthesis parameters
synthesis_parameters = {'harmonic': output_arrays, 'non-harmonic': opening}
synthesis_parameters_path = Path('synthesis_parameters.pickle')
save_pickle(synthesis_parameters_path, synthesis_parameters)

# Synthesis
start = time()
noise_normalization = 'max'
synthesized_harmonic, time_harmonic = synthesize_from_arrays(output_arrays, duration_synth, fs)
synthesized_non_harmonic, time_noise, white_noise_stft, filtered_noise_stft = synthesize_noise_mask(opening,
                                                                                                    duration_synth,
                                                                                                    noise_normalization,
                                                                                                    **stft_parameters)
synthesized = synthesized_harmonic + synthesized_non_harmonic

white_noise_db = 10 * np.log10(np.abs(white_noise_stft)**2 + eps)
filtered_noise_db = 10 * np.log10(np.abs(filtered_noise_stft)**2 + eps)
print('Time to synthesize: %.3f' % (time() - start))

# Spectrogram of synthesis
start = time()
_, _, stft_resynth = sig.stft(synthesized, **stft_parameters)
print('Time to STFT of output: %.3f' % (time() - start))

spectrogram_resynth = np.abs(stft_resynth)**2
spectrogram_db_resynth = 10 * np.log10(spectrogram_resynth + eps)

# Write audio
original_path = Path('original.wav')
harmonic_path = Path('harmonic.wav')
non_harmonic_path = Path('non-harmonic.wav')
resynthesized_path = Path('resynthesized.wav')

wav.write(original_path, fs, signal.astype(np.float32))
wav.write(harmonic_path, fs, synthesized_harmonic.astype(np.float32))
wav.write(non_harmonic_path, fs, synthesized_non_harmonic.astype(np.float32))
wav.write(resynthesized_path, fs, synthesized.astype(np.float32))

# Plot
if plot:
    start = time()

    # Input
    fig = plot_time_frequency(spectrogram_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_input.eps', bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.savefig('figure_input.png', pad_inches=0, transparent=False)

    # Closing
    fig = plot_time_frequency_2(spectrogram_db, closing, tau, omega, v_min=-120, v_max=0, resolution='s',
                                time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_closing.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # # Erosion
    # fig = plot_time_frequency_2(closing, erosion, tau, omega, v_min=-120, v_max=0, resolution='s',
    #                             time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    # fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    # fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    # plt.tight_layout()
    # # plt.savefig('figure_erosion.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Binary top-hat
    fig = plot_time_frequency(top_hat_binary, tau, omega, v_min=0, v_max=1, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()

    # Top-hat vs closing
    fig = plot_time_frequency_top_hat(closing, top_hat_binary, tau, omega, v_min=-120, v_max=0, resolution='s',
                                      time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size,
                                      show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_top-hat.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Skeleton
    fig = plot_time_frequency_2(top_hat_binary, top_hat_skeleteon, tau, omega, v_min=0, v_max=1, resolution='s',
                                time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()

    # # Top-hat vs closed
    # fig = plot_time_frequency_top_hat(top_hat, top_hat_closed, tau, omega, v_min=0, v_max=20, resolution='s',
    #                                   time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size,
    #                                   show=False)
    # fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    # fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    # plt.tight_layout()

    # # Top-hat vs output
    # fig = plot_time_frequency_top_hat(spectrogram_db_resynth, top_hat_closed, tau, omega, v_min=-120, v_max=0,
    #                                   resolution='s',
    #                                   time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size,
    #                                   show=False)
    # fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    # fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    # plt.tight_layout()

    # # Top-hat alone
    # fig = plot_time_frequency(top_hat_closed, tau, omega, v_min=0, v_max=20, resolution='s',
    #                           time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    # fig.axes[0].set_xlim([0. / time_resolution, duration / time_resolution])
    # fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    # plt.tight_layout()

    # Opening
    fig = plot_time_frequency(opening, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_opening.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # # White noise
    # fig = plot_time_frequency(white_noise_db, tau, omega, v_min=-120, v_max=0, resolution='s',
    #                           time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    # fig.axes[0].set_xlim([0.8 / time_resolution, 5.5 / time_resolution])
    # fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    # plt.tight_layout()
    # # plt.savefig('figure_white_noise.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Filtered noise
    fig = plot_time_frequency(filtered_noise_db, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_filtered_noise.eps', bbox_inches='tight', pad_inches=0, transparent=True)

    # Output
    fig = plot_time_frequency(spectrogram_db_resynth, tau, omega, v_min=-120, v_max=0, resolution='s',
                              time_label='Temps (s)', freq_label='Fréquence (Hz)', fig_size=fig_size, show=False)
    fig.axes[0].set_xlim([0. / time_resolution, duration_synth / time_resolution])
    fig.axes[0].set_ylim([0. * (t_fft * padding_factor), 10000. * (t_fft * padding_factor)])
    plt.tight_layout()
    # plt.savefig('figure_output.eps', bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.savefig('figure_output.png', pad_inches=0, transparent=False)

    print('Time to plot and save: %.3f' % (time() - start))

    # Show
    if show:
        plt.show()


if __name__ == '__main__':
    pass
