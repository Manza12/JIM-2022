import os
from morphology import morphology_top_hat, morphology_top_hat_simplified
from synthesis import recover_vectors_no_tqdm, synthesize_from_arrays
import pickle
import os.path as path
import torch
from pathlib import Path
from audio import audio_main
from morphology import morphology_closing
from spectrogram import spectrogram_main
from synthesis import synthesize_noise_from_image
import numpy as np
import scipy.io.wavfile as wav


def extract_harmonic(name):
    ## PARAMETERS
    # Input
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
    v_min = -140.
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
    output_arrays = recover_vectors_no_tqdm(spectrograms_for_synth, time_vector, frequency_vector,
                                            frequency_steps_erosion, time_resolution, threshold_amplitude,
                                            threshold_duration)

    synthesized_signal, time_array = synthesize_from_arrays(output_arrays, time_vector, fs)

    # Spectrogram output
    synthesized_signal_tensor = torch.tensor(synthesized_signal, device=device, dtype=torch.float32)
    spectrogram_synth_db, _, _, _ = spectrogram_main(synthesized_signal_tensor, fs, n_fft, win_length, hop_length,
                                                     time_resolution, window, device, eps, padding_factor,
                                                     frequency_precision)

    ## SAVE ##
    output_path = Path('output') / Path(name)

    if not path.exists(output_path):
        os.makedirs(output_path)

    # Write audio
    wav.write(output_path / Path(name + '-harmonic.wav'), fs, synthesized_signal.astype(np.float32))
    wav.write(output_path / Path(name + '.wav'), fs, signal.astype(np.float32))

    # Save parameters
    with open(output_path / Path(name + '.pickle'), "wb") as fp:
        pickle.dump(output_arrays, fp)

    # Write numpy arrays
    np.save(output_path / Path(name + '-input' + '.npy'), spectrogram_db.cpu().numpy())
    np.save(output_path / Path(name + '-closing' + '.npy'), spectrogram_db_closing)
    np.save(output_path / Path(name + '-erosion' + '.npy'), spectrograms_db_erosion_np)
    np.save(output_path / Path(name + '-top_hat' + '.npy'), spectrograms_db_top_hat_np)
    np.save(output_path / Path(name + '-output' + '.npy'), spectrograms_db_output_np)
    np.save(output_path / Path(name + '-synth' + '.npy'), spectrogram_synth_db.cpu().numpy())


def extract_inharmonic(name):
    ## PARAMETERS
    # Input
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

    spectrogram_db_output = spectrograms['Output']

    ## Synthesis ##
    white_noise_db, filtered_noise_db, filtered_noise_numpy = synthesize_noise_from_image(len(signal), device,
                                                                                          stft_layer,
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
