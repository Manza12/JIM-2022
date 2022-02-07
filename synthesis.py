import numpy as np
import scipy.signal as sig
import torch
from tqdm import tqdm


def recover_vectors(spectrograms, time_vector, stft_frequencies, frequency_steps_erosion, time_resolution,
                    threshold_amplitude, threshold_duration):
    threshold_duration_bins = int(threshold_duration / time_resolution)

    resulting_arrays = list()

    for k in tqdm(range(spectrograms.shape[1] - 2)):
        frequencies = list()
        timestamps = list()
        amplitudes = list()

        if np.any(spectrograms[:, k, :] > threshold_amplitude):
            on = False
            for n in range(spectrograms.shape[2]):
                neighborhood = spectrograms[:, k: k + 3, n]
                amp_db = np.max(neighborhood)
                idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)

                if amp_db > threshold_amplitude:
                    on = True
                    frequencies += [stft_frequencies[k + idx[1]] - idx[0] * frequency_steps_erosion]
                    timestamps += [time_vector[n]]
                    amplitudes += [10 ** (amp_db / 20)]

                    spectrograms[:, k: k + 3, n] = threshold_amplitude
                else:
                    if on:
                        if len(frequencies) > threshold_duration_bins:
                            resulting_array = np.array((frequencies, timestamps, amplitudes))
                            resulting_arrays.append(resulting_array)
                        frequencies = list()
                        timestamps = list()
                        amplitudes = list()
                    on = False
            if on:
                if len(frequencies) > threshold_duration_bins:
                    resulting_array = np.array((frequencies, timestamps, amplitudes))
                    resulting_arrays.append(resulting_array)

    return resulting_arrays


def recover_vectors_bis(spectrogram, time_vector, stft_frequencies, time_resolution, neighbourhood_width,
                        threshold_amplitude, threshold_duration):
    threshold_duration_bins = int(threshold_duration / time_resolution)

    resulting_arrays = list()

    for k in tqdm(range(spectrogram.shape[0] - neighbourhood_width)):
        frequencies = list()
        timestamps = list()
        amplitudes = list()

        if np.any(spectrogram[k, :] > threshold_amplitude):
            on = False
            for n in range(spectrogram.shape[1]):
                neighborhood = spectrogram[k: k + neighbourhood_width, n]
                amp_db = np.max(neighborhood)
                idx = np.argmax(neighborhood)

                if amp_db > threshold_amplitude:
                    on = True
                    frequencies += [stft_frequencies[k + idx]]
                    timestamps += [time_vector[n]]
                    amplitudes += [2 * 10 ** (amp_db / 20)]

                    spectrogram[k: k + 3, n] = threshold_amplitude
                else:
                    if on:
                        if len(frequencies) > threshold_duration_bins:
                            resulting_array = np.array((frequencies, timestamps, amplitudes))
                            resulting_arrays.append(resulting_array)
                        frequencies = list()
                        timestamps = list()
                        amplitudes = list()
                    on = False
            if on:
                if len(frequencies) > threshold_duration_bins:
                    resulting_array = np.array((frequencies, timestamps, amplitudes))
                    resulting_arrays.append(resulting_array)

    return resulting_arrays


def recover_vectors_no_tqdm(spectrograms, time_vector, stft_frequencies, frequency_steps_erosion, time_resolution,
                            threshold_amplitude, threshold_duration):
    threshold_duration_bins = int(threshold_duration / time_resolution)

    resulting_arrays = list()

    for k in range(spectrograms.shape[1] - 2):
        frequencies = list()
        timestamps = list()
        amplitudes = list()

        if np.any(spectrograms[:, k, :] > threshold_amplitude):
            on = False
            for n in range(spectrograms.shape[2]):
                neighborhood = spectrograms[:, k: k + 3, n]
                amp_db = np.max(neighborhood)
                idx = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)

                if amp_db > threshold_amplitude:
                    on = True
                    frequencies += [stft_frequencies[k + idx[1]] - idx[0] * frequency_steps_erosion]
                    timestamps += [time_vector[n]]
                    amplitudes += [10 ** (amp_db / 20)]

                    spectrograms[:, k: k + 3, n] = threshold_amplitude
                else:
                    if on:
                        if len(frequencies) > threshold_duration_bins:
                            resulting_array = np.array((frequencies, timestamps, amplitudes))
                            resulting_arrays.append(resulting_array)
                        frequencies = list()
                        timestamps = list()
                        amplitudes = list()
                    on = False
            if on:
                if len(frequencies) > threshold_duration_bins:
                    resulting_array = np.array((frequencies, timestamps, amplitudes))
                    resulting_arrays.append(resulting_array)

    return resulting_arrays


def phase_vocoder(frequencies, amplitudes, fs):
    phase = 2 * np.pi * np.cumsum(frequencies / fs)
    signal = amplitudes * np.sin(phase)
    return signal


def synthesize_from_arrays(resulting_arrays, duration, fs):
    time_array = np.arange(int(fs * duration)) / fs
    synthesized_signal = np.zeros_like(time_array)

    for resulting_array in resulting_arrays:
        frequencies = resulting_array[0, :]
        timestamps = resulting_array[1, :]
        amplitudes = resulting_array[2, :]

        frequencies = np.concatenate((np.zeros(1), frequencies[:-1]))
        amplitudes = np.concatenate((np.zeros(1), amplitudes[:-1]))

        n_start = int(fs * timestamps[0])
        n_end = int(fs * timestamps[-1])
        amplitudes_itp = interpolate(time_array[n_start: n_end], timestamps, amplitudes, interpolation_type='linear')
        frequencies_itp = interpolate(time_array[n_start: n_end], timestamps, frequencies, interpolation_type='linear')
        np.interp(time_array[n_start: n_end], timestamps, frequencies)

        synthesized_signal[n_start: n_end] += phase_vocoder(frequencies_itp, amplitudes_itp, fs)

    return synthesized_signal, time_array


def synthesize_noise_from_image(signal_size, device, stft_layer, eps, spectrogram_output_db, sigma=10.):
    # Create noise
    white_noise = torch.randn(signal_size, device=device) * sigma
    white_noise_spectrogram = stft_layer(white_noise)[0]
    white_noise_power = white_noise_spectrogram[:, :, 0].pow(2) + white_noise_spectrogram[:, :, 1].pow(2)
    white_noise_db = 10 * torch.log10(white_noise_power + eps ** 2)

    # Filter spectrogram
    filtered_noise_spectrogram = white_noise_spectrogram * 10 ** (spectrogram_output_db.unsqueeze(-1) / 20)
    filtered_noise_power = filtered_noise_spectrogram[:, :, 0].pow(2) + filtered_noise_spectrogram[:, :, 1].pow(2)
    filtered_noise_db = 10 * torch.log10(filtered_noise_power + eps ** 2)

    # Synthesize
    filtered_noise = stft_layer.inverse(filtered_noise_spectrogram.unsqueeze(0))
    filtered_noise_numpy = filtered_noise[0].cpu().numpy()

    return white_noise_db, filtered_noise_db, filtered_noise_numpy


def interpolate(time_array, timestamps, y, interpolation_type='linear'):
    if interpolation_type == 'linear':
        return np.interp(time_array, timestamps, y)
    elif interpolation_type == 'fourier':
        return sig.resample(y, len(time_array))

