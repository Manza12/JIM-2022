import sys
from pathlib import Path; sys.path.insert(0, str(Path('..') / Path('..') / Path('..')))
import scipy.io.wavfile as wav
import scipy.signal as sig
from utils import pad_and_cut
from plot import plot_figures
import numpy as np
from time import time
import scipy.ndimage.morphology as morpho
from synthesis import synthesize_from_arrays, synthesize_noise_mask, path_following
from utils import save_pickle
from skeleton import skeleton

plot = False
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
window_size = int(fs * t_fft)  # in samples
padding_factor = 2
n_fft = window_size * padding_factor  # in samples
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
top_hat_skeleton = skeleton(top_hat_binary)

# Create detector food image
detector_food = np.copy(top_hat_skeleton)

# Opening
opening_frequency_width = 100  # in Hertz
opening_width = int(opening_frequency_width / frequency_precision)  # in samples

str_el_ope = np.zeros((opening_width, 1))

opening = morpho.grey_opening(closing, structure=str_el_ope)

# Erosion
erosion_time_width = int(t_fft / time_resolution)
window_shape = sig.windows.get_window(window, erosion_time_width)
erosion_shape = 10 * np.log10(window_shape + eps)
str_el_ero = np.expand_dims(erosion_shape, 0)

erosion = morpho.grey_erosion(opening, structure=str_el_ero)

print('Time to morphology: %.3f' % (time() - start))

# Recover parameters
threshold_duration = 0.
linking_time = 0.
neighbourhood_width = top_hat_width

start = time()
output_arrays = path_following(closing, detector_food, tau, omega, neighbourhood_width,
                               threshold_duration, linking_time, time_resolution)
print('Time to recover vectors: %.3f' % (time() - start))

# Synthesis
start = time()
noise_normalization = 'morphological'
size = 20
synthesized_harmonic, time_harmonic = synthesize_from_arrays(output_arrays, duration_synth, fs)
synthesized_non_harmonic, time_noise, white_noise_stft, filtered_noise_stft = synthesize_noise_mask(erosion,
                                                                                                    duration_synth,
                                                                                                    noise_normalization,
                                                                                                    size=size,
                                                                                                    **stft_parameters,
                                                                                                    )
synthesized = synthesized_harmonic + synthesized_non_harmonic[:len(synthesized_harmonic)]

white_noise_db = 10 * np.log10(np.abs(white_noise_stft)**2 + eps)
filtered_noise_db = 10 * np.log10(np.abs(filtered_noise_stft)**2 + eps)
print('Time to synthesize: %.3f' % (time() - start))

# Spectrogram of synthesis
start = time()
_, _, stft_resynth = sig.stft(synthesized, **stft_parameters)
_, _, stft_resynth_harmonic = sig.stft(synthesized_harmonic, **stft_parameters)
_, _, stft_resynth_non_harmonic = sig.stft(synthesized_non_harmonic, **stft_parameters)
print('Time to STFT of output: %.3f' % (time() - start))

spectrogram_resynth = np.abs(stft_resynth)**2
spectrogram_resynth_harmonic = np.abs(stft_resynth_harmonic)**2
spectrogram_resynth_non_harmonic = np.abs(stft_resynth_non_harmonic)**2

spectrogram_output = 10 * np.log10(spectrogram_resynth + eps)
spectrogram_resynth_harmonic_db = 10 * np.log10(spectrogram_resynth_harmonic + eps)
spectrogram_resynth_non_harmonic_db = 10 * np.log10(spectrogram_resynth_non_harmonic + eps)

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
np.save(str(arrays_path / 'top_hat_skeleton.npy'), top_hat_skeleton)
np.save(str(arrays_path / 'opening.npy'), opening)
np.save(str(arrays_path / 'erosion.npy'), erosion)
np.save(str(arrays_path / 'white_noise_db.npy'), white_noise_db)
np.save(str(arrays_path / 'filtered_noise_db.npy'), filtered_noise_db)
np.save(str(arrays_path / 'spectrogram_output.npy'), spectrogram_output)
np.save(str(arrays_path / 'spectrogram_resynth_harmonic.npy'), spectrogram_resynth_harmonic_db)
np.save(str(arrays_path / 'spectrogram_resynth_non-harmonic.npy'), spectrogram_resynth_non_harmonic_db)


# Plot
if plot:
    start = time()
    plot_figures(Path('.'), fig_size=fig_size)
    print('Time to plot and save: %.3f' % (time() - start))


if __name__ == '__main__':
    pass
