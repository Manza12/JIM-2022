import sys
from pathlib import Path; sys.path.insert(0, str(Path('..') / Path('..') / Path('..')))
import scipy.io.wavfile as wav
import scipy.signal as sig
from utils import pad_and_cut
from plot import plot_figures_paper
import numpy as np
import matplotlib.pyplot as plt
from time import time
import scipy.ndimage.morphology as morpho
from synthesis import synthesize_from_arrays, synthesize_noise_mask, path_following
from utils import save_pickle
from skeleton import skeleton

plot = True
fig_size = (640, 360)

# Input
names_list = Path('.').absolute().as_posix().split('/')
note = names_list[-1]
instrument = names_list[-2]
signal_path = Path(instrument + '_' + note + '.wav')

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
spectrogram_input = 10 * np.log10(spectrogram + eps)

# Morphology
start = time()

# Closing
closing_time_width = 0.05  # in seconds
closing_frequency_width = 50.  # in Hertz
closing_shape = (int(closing_frequency_width / frequency_precision), int(closing_time_width / time_resolution))
str_el_clo = np.zeros(closing_shape)

closing = morpho.grey_closing(spectrogram_input, structure=str_el_clo)

# Top-hat
top_hat_width = 5  # in samples
top_hat_freq_width = top_hat_width * frequency_precision  # in Hertz
threshold = 3.  # in dB

str_el_top_hat = np.zeros((top_hat_width, 1))

top_hat = closing - morpho.grey_opening(closing, structure=str_el_top_hat)

# Binarize top-hat
top_hat_binary = np.zeros_like(closing, dtype=np.int8)
top_hat_binary[top_hat > threshold] = 1

# Skeletonize the top-hat
str_el_skull_1 = np.ones((3, 1), dtype=bool)
str_el_skull_2 = np.ones((2, 1), dtype=bool)

top_hat_skeleteon = skeleton(top_hat_binary, str_el_skull_1)
top_hat_skeleteon = skeleton(top_hat_skeleteon, str_el_skull_2)

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
neighbourhood_width = top_hat_width

start = time()
output_arrays = path_following(closing, detector_food, tau, omega, neighbourhood_width,
                               threshold_duration, linking_time, time_resolution)
print('Time to recover vectors: %.3f' % (time() - start))

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
spectrogram_output = 10 * np.log10(spectrogram_resynth + eps)

# Write parameters
parameters_path = Path('parameters')
parameters_path.mkdir(parents=True, exist_ok=True)

synthesis_parameters = {'harmonic': output_arrays, 'non-harmonic': opening}
synthesis_parameters_path = parameters_path / Path('synthesis_parameters.pickle')
save_pickle(synthesis_parameters_path, synthesis_parameters)

plot_parameters = {'time_resolution': time_resolution, 'duration_synth': duration_synth, 't_fft': t_fft,
                   'padding_factor': padding_factor}
plot_parameters_path = parameters_path / Path('plot_parameters.pickle')
save_pickle(plot_parameters_path, plot_parameters)

# Write audio
audio_path = Path('audio')
audio_path.mkdir(parents=True, exist_ok=True)

original_path = audio_path / Path('input.wav')
harmonic_path = audio_path / Path('harmonic.wav')
non_harmonic_path = audio_path / Path('non-harmonic.wav')
resynthesized_path = audio_path / Path('output.wav')

wav.write(original_path, fs, signal.astype(np.float32))
wav.write(harmonic_path, fs, synthesized_harmonic.astype(np.float32))
wav.write(non_harmonic_path, fs, synthesized_non_harmonic.astype(np.float32))
wav.write(resynthesized_path, fs, synthesized.astype(np.float32))

# Write arrays
arrays_path = Path('arrays')
arrays_path.mkdir(parents=True, exist_ok=True)

np.save(str(arrays_path / 'tau.npy'), tau)
np.save(str(arrays_path / 'omega.npy'), omega)

np.save(str(arrays_path / 'spectrogram_input.npy'), spectrogram_input)
np.save(str(arrays_path / 'closing.npy'), closing)
np.save(str(arrays_path / 'top_hat_binary.npy'), top_hat_binary)
np.save(str(arrays_path / 'top_hat_skeleteon.npy'), top_hat_skeleteon)
np.save(str(arrays_path / 'opening.npy'), opening)
np.save(str(arrays_path / 'filtered_noise_db.npy'), filtered_noise_db)
np.save(str(arrays_path / 'spectrogram_output.npy'), spectrogram_output)


# Plot
if plot:
    start = time()
    plot_figures_paper(Path('.'), fig_size=fig_size, save=True)
    print('Time to plot and save: %.3f' % (time() - start))


if __name__ == '__main__':
    pass
