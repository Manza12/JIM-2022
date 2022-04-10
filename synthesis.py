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

    for k in range(spectrogram.shape[0] - neighbourhood_width):
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

                    spectrogram[k: k + neighbourhood_width, n] = threshold_amplitude
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


def split_in_connected_components(non_minimal_frequencies):
    connected_components = []

    connected_component = []
    for k, idx in enumerate(non_minimal_frequencies):
        if k == 0:
            connected_component.append(idx)
        else:
            if idx - non_minimal_frequencies[k-1] == 1:
                connected_component.append(idx)
            else:
                connected_components.append(connected_component)
                connected_component = [idx]
        if k == len(non_minimal_frequencies) - 1:
            if len(connected_component) != 0:
                connected_components.append(connected_component)
    return connected_components


def pick_starting_frequencies(connected_components, n, spectrogram):
    starting_frequencies = []
    for connected_component in connected_components:
        min_idx = min(connected_component)
        max_idx = max(connected_component)
        starting_frequency = np.argmax(spectrogram[min_idx: max_idx + 1, n]) + min_idx
        starting_frequencies.append(starting_frequency)
    return starting_frequencies


def follow_the_ridge(spectrogram, n, starting_frequency, low_width, high_width, minimal_value,
                     time_vector, stft_frequencies, linking_bins):
    timestamps = [time_vector[n]]
    frequencies = [stft_frequencies[starting_frequency]]
    amplitudes = [2 * 10 ** (spectrogram[starting_frequency, n] / 20)]

    current_time_index = n
    current_frequency_index = starting_frequency

    current_linking_tolerance = 0

    while current_time_index + 1 < spectrogram.shape[1]:
        # Put current spectrogram slice to minimal value
        spectrogram[current_frequency_index - low_width: current_frequency_index + high_width, current_time_index] = \
            minimal_value

        # Get new slice
        spectrogram_slice = spectrogram[
                            current_frequency_index - low_width: current_frequency_index + high_width,
                            current_time_index + 1
                            ]

        amplitude_value = spectrogram_slice.max()
        if amplitude_value > minimal_value:
            amplitudes.append(2 * 10 ** (amplitude_value / 20))

            next_frequency = np.argmax(spectrogram_slice) + current_frequency_index - low_width

            timestamps.append(time_vector[current_time_index + 1])
            frequencies.append(stft_frequencies[next_frequency])
        else:
            current_linking_tolerance += 1

            if current_linking_tolerance > linking_bins:
                break
            else:
                next_frequency = current_frequency_index

        # Update
        current_frequency_index = next_frequency
        current_time_index += 1

    return timestamps, frequencies, amplitudes


def ridge_following(spectrogram, time_vector, stft_frequencies, neighbourhood_width,
                    threshold_duration, linking_time, time_resolution):
    resulting_arrays = []

    threshold_duration_bins = int(threshold_duration / time_resolution)
    linking_bins = int(linking_time / time_resolution)

    minimal_value = spectrogram.min()

    low_width = neighbourhood_width - neighbourhood_width // 2 - 1
    high_width = neighbourhood_width - low_width
    for n in range(spectrogram.shape[1]):
        if np.any(spectrogram[:, n] > minimal_value):
            non_minimal_frequencies = np.where(spectrogram[:, n] > minimal_value)[0]
            connected_components = split_in_connected_components(non_minimal_frequencies)
            starting_frequencies = pick_starting_frequencies(connected_components, n, spectrogram)
            for starting_frequency in starting_frequencies:
                # if 740 < starting_frequency < 760:
                #     print('')
                timestamps, frequencies, amplitudes = \
                    follow_the_ridge(spectrogram, n, starting_frequency, low_width, high_width, minimal_value,
                                     time_vector, stft_frequencies, linking_bins)
                if len(frequencies) > threshold_duration_bins:
                    resulting_array = np.array((frequencies, timestamps, amplitudes))
                    resulting_arrays.append(resulting_array)

    return resulting_arrays


def follow_the_path(spectrogram, mask, n, starting_frequency, low_width, high_width,
                    time_vector, stft_frequencies, linking_bins):
    timestamps = [time_vector[n]]
    frequencies = [stft_frequencies[starting_frequency]]
    amplitudes = [2 * 10 ** (spectrogram[starting_frequency, n] / 20)]

    current_time_index = n
    current_frequency_index = starting_frequency

    current_linking_tolerance = 0

    while current_time_index + 1 < spectrogram.shape[1]:
        # Put current mask slice to 0
        mask[current_frequency_index - low_width: current_frequency_index + high_width, current_time_index] = 0

        # Get new slice
        mask_slice = mask[
                            current_frequency_index - low_width: current_frequency_index + high_width,
                            current_time_index + 1
                            ]

        if np.any(mask_slice) > 0:
            index = np.where(mask_slice > 0)[0]
            assert index.shape[0] == 1
            index = index[0]

            next_frequency_index = index + current_frequency_index - low_width

            amplitude_value = spectrogram[next_frequency_index, current_time_index]

            amplitudes.append(2 * 10 ** (amplitude_value / 20))
            timestamps.append(time_vector[current_time_index + 1])
            frequencies.append(stft_frequencies[next_frequency_index])
        else:
            current_linking_tolerance += 1

            if current_linking_tolerance > linking_bins:
                break
            else:
                next_frequency_index = current_frequency_index

        # Update
        current_frequency_index = next_frequency_index
        current_time_index += 1

    return timestamps, frequencies, amplitudes


def path_following(spectrogram, mask, time_vector, stft_frequencies, neighbourhood_width,
                   threshold_duration, linking_time, time_resolution):
    resulting_arrays = []

    threshold_duration_bins = int(threshold_duration / time_resolution)
    linking_bins = int(linking_time / time_resolution)

    low_width = neighbourhood_width - neighbourhood_width // 2 - 1
    high_width = neighbourhood_width - low_width
    for n in range(spectrogram.shape[1]):
        if np.any(mask[:, n] > 0):
            starting_frequencies = np.where(mask[:, n] > 0)[0]
            for starting_frequency in starting_frequencies:
                timestamps, frequencies, amplitudes = \
                    follow_the_path(spectrogram, mask, n, starting_frequency, low_width, high_width,
                                    time_vector, stft_frequencies, linking_bins)
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


def synthesize_noise_mask(mask, duration, normalization='max', **stft_parameters):
    N = int(duration * stft_parameters['fs'])

    # Create noise
    white_noise = np.random.rand(N) * 2 - 1
    omega, tau, white_noise_stft = sig.stft(white_noise, **stft_parameters)

    # Normalize noise
    if normalization == 'mean':
        log_mean = np.mean(10 * np.log10(np.abs(white_noise_stft) ** 2))
        white_noise_normalized_stft = white_noise_stft / 10 ** (log_mean / 20)
    elif normalization == 'max':
        log_max = np.max(10 * np.log10(np.abs(white_noise_stft) ** 2))
        white_noise_normalized_stft = white_noise_stft / 10**(log_max / 20)
    else:
        white_noise_normalized_stft = white_noise_stft

    # Filter spectrogram
    filtered_noise_stft = white_noise_normalized_stft * 10 ** (mask / 20)

    # Synthesize
    time_array, filtered_noise = sig.istft(filtered_noise_stft, **stft_parameters)

    return filtered_noise, time_array, white_noise_normalized_stft, filtered_noise_stft


def interpolate(time_array, timestamps, y, interpolation_type='linear'):
    if interpolation_type == 'linear':
        return np.interp(time_array, timestamps, y)
    elif interpolation_type == 'fourier':
        return sig.resample(y, len(time_array))

