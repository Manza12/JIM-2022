import torch
from matplotlib.pyplot import show
from pathlib import Path
from audio import audio_main
from morphology import morphology_closing
from plot import plot_signal, plot_time_frequency
from spectrogram import spectrogram_main
from synthesis import synthesize_noise_from_image
import numpy as np
import scipy.io.wavfile as wav
import os
import os.path as path


## PARAMETERS
# Input
name = '33'
extension = '.wav'
folder = 'samples'
file_path = Path(folder) / Path(name + extension)
starting_rest = 1.
final_fade = 1.
duration = 10.

# STFT
fs = 44100
ts = 1 / fs
time_resolution = 0.01
t_fft = 0.1  # seconds
n_fft = 4410
padding_factor = 1
frequency_precision = 1 / t_fft  # in Hertz
win_length = int(t_fft * fs)
hop_length = int(fs * time_resolution)
window = 'blackman'
device = 'cuda:0'

# Spectrogram Plot
plot = True
v_min = -140.
eps = 10**(v_min/10)

# Morphology
closing_time_width = 0.05  # in seconds
closing_frequency_width = 50.  # in Hertz

opening_time_width = 0.  # in seconds
opening_frequency_width = 100  # in Hertz

## AUDIO
signal, signal_tensor, t = audio_main(file_path, fs, starting_rest, duration, final_fade, ts, device)

## Spectrogram
spectrogram_db, frequency_vector, time_vector, stft_layer = spectrogram_main(signal_tensor, fs, n_fft, win_length,
                                                                             hop_length,
                                                                             time_resolution, window, device, eps,
                                                                             padding_factor, frequency_precision)

## MORPHOLOGY
spectrograms = morphology_closing(spectrogram_db, device, time_resolution, frequency_precision,
                                  closing_frequency_width, closing_time_width, opening_frequency_width,
                                  opening_time_width)

spectrogram_db_closing = spectrograms['Closing']
spectrogram_db_output = spectrograms['Output']

## Synthesis ##
white_noise_db, filtered_noise_db, filtered_noise_numpy = synthesize_noise_from_image(len(signal), device, stft_layer,
                                                                                      eps, spectrogram_db_output,
                                                                                      sigma=10.)

# Spectrogram output
synthesized_signal_tensor = torch.tensor(filtered_noise_numpy, device=device, dtype=torch.float32)
spectrogram_synth_db, _, _, _ = spectrogram_main(synthesized_signal_tensor, fs, n_fft, win_length, hop_length,
                                                 time_resolution, window, device, eps, padding_factor,
                                                 frequency_precision)

## SAVE ##
# Create folder
output_path = Path('output') / Path(name)

if not path.exists(output_path):
    os.makedirs(output_path)

# Write audio
wav.write(output_path / Path(name + '-inharmonic.wav'), fs, filtered_noise_numpy.astype(np.float32))

# # Write numpy arrays
np.save(output_path / Path(name + '-filter_shape' + '.npy'), spectrogram_db_output.cpu().numpy())


## PLOTS
if plot:
    # Plot audio
    fig_signal = plot_signal(signal, t)

    # Plot spectrogram
    fig_input = plot_time_frequency(spectrogram_db, time_vector, frequency_vector, v_min=v_min, v_max=0, c_map='Greys',
                                    resolution='ms', fig_title='Input', show=False, numpy=False, interpolation='none',
                                    full_screen=True)

    fig_closing = plot_time_frequency(spectrogram_db_closing, time_vector, frequency_vector, v_min=v_min, v_max=0,
                                      c_map='Greys', resolution='ms', fig_title='Closing',
                                      show=False, numpy=False, full_screen=True, interpolation='none')

    fig_opening = plot_time_frequency(spectrogram_db_output, time_vector, frequency_vector, v_min=v_min, v_max=0,
                                      c_map='Greys', resolution='ms', fig_title='Closing',
                                      show=False, numpy=False, full_screen=True, interpolation='none')

    fig_synth = plot_time_frequency(spectrogram_synth_db, time_vector, frequency_vector, v_min=v_min, v_max=0,
                                    c_map='Greys', resolution='ms', fig_title='Synthesized', interpolation='none',
                                    show=False, numpy=False, full_screen=True)

    show(block=True)

if __name__ == '__main__':
    pass
