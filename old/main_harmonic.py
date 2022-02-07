import os
import torch
from matplotlib.pyplot import show
from pathlib import Path
from audio import audio_main
from morphology import morphology_top_hat, morphology_top_hat_simplified
from plot import plot_signal, plot_time_frequency, plot_slider, update_slider, plot_harmonics
from spectrogram import spectrogram_main
from synthesis import recover_vectors, synthesize_from_arrays
import pickle
import numpy as np
import scipy.io.wavfile as wav
from scipy.ndimage.morphology import grey_dilation
import os.path as path


## PARAMETERS
# Input
name = '45'
extension = '.wav'
folder = 'samples'
file_path = Path(folder) / Path(name + extension)
starting_rest = 1.
final_fade = 1.
duration = 10.

# STFT
fs = 44100
ts = 1 / fs
time_resolution = 0.001
t_fft = 0.1  # seconds
n_fft = 4410
padding_factor = 2
frequency_precision = 1 / t_fft  # in Hertz
win_length = int(t_fft * fs)
hop_length = int(fs * time_resolution)
window = 'blackman'
device = 'cuda:0'

# Spectrogram Plot
plot = False
v_min = -140.
v_max = 0.
eps = 10**(v_min/10)

# Morphology
simple = True

closing_time_width = 0.05  # in seconds
closing_frequency_width = 50.  # in Hertz

n_freq_bins_erosion = 5 * padding_factor
frequency_steps_erosion = 1.  # Hertz
factor = int(frequency_precision / frequency_steps_erosion)

top_hat_width = 5

threshold = 10.  # dB

closing_time_width_final = 0.15  # in seconds

# Synthesis
threshold_amplitude = v_min
threshold_duration = 0.05  # in seconds

## AUDIO
signal, signal_tensor, t = audio_main(file_path, fs, starting_rest, duration, final_fade, ts, device)

## Spectrogram
spectrogram_db, frequency_vector, time_vector, stft_layer = spectrogram_main(signal_tensor, fs, n_fft, win_length,
                                                                             hop_length,
                                                                             time_resolution, window, device, eps,
                                                                             padding_factor, frequency_precision)

## MORPHOLOGY
if simple:
    spectrograms = morphology_top_hat_simplified(spectrogram_db, window, fs, n_fft, eps, device, time_resolution,
                                                 frequency_precision, closing_frequency_width, closing_time_width,
                                                 n_freq_bins_erosion, threshold, padding_factor,
                                                 closing_time_width_final, top_hat_width)
else:
    spectrograms = morphology_top_hat(spectrogram_db, window, fs, n_fft, eps, device, time_resolution,
                                      frequency_precision, closing_frequency_width, closing_time_width, factor,
                                      n_freq_bins_erosion, threshold, closing_time_width_final)

spectrogram_db_closing = spectrograms['Closing']
spectrograms_db_erosion_np = spectrograms['Erosion']
spectrograms_db_top_hat_np = spectrograms['Top-hat']
spectrograms_db_output_np = spectrograms['Output']

## Synthesis ##
spectrograms_for_synth = np.copy(spectrograms_db_output_np)
output_arrays = recover_vectors(spectrograms_for_synth, time_vector, frequency_vector, frequency_steps_erosion,
                                time_resolution, threshold_amplitude, threshold_duration)

synthesized_signal, time_array = synthesize_from_arrays(output_arrays, time_vector[-1], fs)

# Spectrogram output
synthesized_signal_tensor = torch.tensor(synthesized_signal, device=device, dtype=torch.float32)
spectrogram_synth_db, _, _, _ = spectrogram_main(synthesized_signal_tensor, fs, n_fft, win_length, hop_length,
                                                 time_resolution, window, device, eps, padding_factor,
                                                 frequency_precision)

## SAVE ##
# Create folder
output_path = Path('output') / Path(name)

if not path.exists(output_path):
    os.makedirs(output_path)

# Write audio
wav.write(output_path / Path(name + '-harmonic.wav'), fs, synthesized_signal.astype(np.float32))
wav.write(output_path / Path(name + '.wav'), fs, signal.astype(np.float32))

# Save parameters
with open(Path('output') / Path(name + '.pickle'), "wb") as fp:
    pickle.dump(output_arrays, fp)

# Write numpy arrays
np.save(output_path / Path(name + '-input' + '.npy'), spectrogram_db.cpu().numpy())
np.save(output_path / Path(name + '-closing' + '.npy'), spectrogram_db_closing)
np.save(output_path / Path(name + '-erosion' + '.npy'), spectrograms_db_erosion_np)
np.save(output_path / Path(name + '-top_hat' + '.npy'), spectrograms_db_top_hat_np)
np.save(output_path / Path(name + '-output' + '.npy'), spectrograms_db_output_np)
np.save(output_path / Path(name + '-synth' + '.npy'), spectrogram_synth_db.cpu().numpy())


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
                                      show=False, numpy=True, full_screen=True, interpolation='none')

    if simple:
        fig_erosion = plot_time_frequency(spectrograms_db_erosion_np[0, :, :], time_vector, frequency_vector,
                                          v_min=v_min,
                                          v_max=0, c_map='Greys', resolution='ms', fig_title='Erosion', show=False,
                                          numpy=True, full_screen=True, interpolation='none')

        fig_top_hat = plot_time_frequency(spectrograms_db_top_hat_np[0, :, :], time_vector, frequency_vector,
                                          v_min=0, v_max=20, c_map='Greys', resolution='ms', fig_title='Top-hat',
                                          show=False, numpy=True, full_screen=True, interpolation='none')

        fig_output = plot_time_frequency(grey_dilation(spectrograms_db_output_np[0, :, :], structure=np.zeros((1, 1))),
                                         time_vector, frequency_vector, v_min=v_min, v_max=0, c_map='Greys',
                                         resolution='ms', fig_title='Output', show=False, numpy=True, full_screen=True,
                                         interpolation='none')
    else:
        figure_erosion, image_erosion, slider_erosion = plot_slider(spectrograms_db_erosion_np, frequency_vector,
                                                                    time_vector, title='Erosion',
                                                                    slider_title='Frequency shift',
                                                                    c_map='Greys', block=False, show=False, v_min=v_min,
                                                                    v_max=0, resolution='ms', interpolation='none')

        slider_erosion.on_changed(lambda val: update_slider(val, figure=figure_erosion, image=image_erosion,
                                                            array=spectrograms_db_erosion_np))

        fig_top_hat, image_top_hat, slider_top_hat = plot_slider(spectrograms_db_top_hat_np, frequency_vector,
                                                                 time_vector,
                                                                 title='Top-hat', slider_title='Frequency shift',
                                                                 c_map='Greys', block=False, show=False, v_min=0,
                                                                 v_max=20,
                                                                 resolution='ms', interpolation='none')

        slider_top_hat.on_changed(lambda val: update_slider(val, figure=fig_top_hat, image=image_top_hat,
                                                            array=spectrograms_db_top_hat_np))

        fig_output, image_output, slider_output = plot_slider(spectrograms_db_output_np, frequency_vector, time_vector,
                                                              title='Output', slider_title='Frequency shift',
                                                              c_map='Greys', block=False, show=False, v_min=v_min,
                                                              v_max=0,
                                                              resolution='ms', interpolation='none')

        slider_output.on_changed(lambda val: update_slider(val, figure=fig_output, image=image_output,
                                                           array=spectrograms_db_output_np))

    fig_synth = plot_time_frequency(spectrogram_synth_db, time_vector, frequency_vector, v_min=v_min, v_max=0,
                                    c_map='Greys', resolution='ms', fig_title='Synthesized', interpolation='none',
                                    show=False, numpy=False, full_screen=True)

    plot_harmonics(output_arrays)

    # show(block=True)

if __name__ == '__main__':
    pass
