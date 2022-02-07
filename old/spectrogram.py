from nnAudio.features.stft import STFT
from time import time
import torch
import numpy as np


def spectrogram_main(signal_tensor, fs, n_fft, win_length, hop_length, time_resolution, window, device, eps,
                     padding_factor, frequency_precision):
    # Create layer
    stft_layer = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, pad_mode='constant',
                      iSTFT=True, sr=fs, freq_bins=(n_fft * padding_factor) // 2 + 1, freq_scale='linear',
                      fmin=0, fmax=fs/2-frequency_precision).to(device)

    # Compute spectrogram
    start = time()
    spectrogram = stft_layer(signal_tensor)
    spectrogram_power = spectrogram[0, :, :, 0] ** 2 + spectrogram[0, :, :, 1] ** 2
    spectrogram_db = 10 * torch.log10(spectrogram_power + eps)
    print('Time to compute spectrogram: %.3f' % (time() - start))

    # Recover time-frequency vectors
    frequency_vector = np.array(stft_layer.bins2freq)
    time_vector = np.arange(0, time_resolution * spectrogram_db.shape[1], time_resolution)

    return spectrogram_db, frequency_vector, time_vector, stft_layer
