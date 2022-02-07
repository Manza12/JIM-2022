import numpy as np
import torch
from time import time
from utils import get_str_el
from nnMorpho.operations import erosion, top_hat, closing, opening


def morphology_top_hat(spectrogram_db, window, fs, n_fft, eps, device, time_resolution, frequency_precision,
                       closing_frequency_width, closing_time_width, factor, n_freq_bins_erosion, threshold,
                       closing_time_width_final):
    spectrograms = {'Input': spectrogram_db}

    # Structuring elements
    closing_shape = (int(closing_frequency_width / frequency_precision), int(closing_time_width / time_resolution))
    str_el_clo = torch.zeros(closing_shape, device=device)

    str_els_ero = torch.empty((factor, n_freq_bins_erosion, 1), device=device)
    idx_0 = (- n_freq_bins_erosion // 2 + 1) * factor
    window_spectrum = get_str_el(window, fs, n_fft, factor=factor, plot=False, db=True, eps=eps).astype(np.float32)
    for i in range(factor):
        idx = np.arange(idx_0 + i, idx_0 + n_freq_bins_erosion * factor + i, factor)
        erosion_freq_shape = np.take(window_spectrum, idx)
        str_els_ero[i, :, 0] = torch.tensor(erosion_freq_shape, device=device, dtype=torch.float32)

    str_el_th = torch.zeros((3, 1), device=device)

    str_el_clo_final = torch.zeros((1, int(closing_time_width_final / time_resolution)), device=device)

    # Computations
    start = time()
    spectrogram_db_closing = closing(spectrogram_db, str_el_clo)

    size = (factor, spectrogram_db.shape[0], spectrogram_db.shape[1])
    spectrograms_db_erosion = torch.empty(size, device=device)
    spectrograms_db_top_hat = torch.empty(size, device=device)
    spectrograms_db_output = torch.zeros(size, device=device) - 1000

    for i in range(factor):
        spectrograms_db_erosion[i, :, :] = erosion(spectrogram_db_closing, str_els_ero[i, :, :])
        spectrograms_db_top_hat[i, :, :] = top_hat(spectrograms_db_erosion[i, :, :], str_el_th)
        spectrograms_db_output[i, :, :][spectrograms_db_top_hat[i, :, :] > threshold] = \
            spectrograms_db_erosion[i, :, :][spectrograms_db_top_hat[i, :, :] > threshold]

    spectrogram_db_closing_np = spectrogram_db_closing.cpu().numpy()
    spectrograms_db_erosion_np = spectrograms_db_erosion.cpu().numpy()
    spectrograms_db_top_hat_np = spectrograms_db_top_hat.cpu().numpy()
    spectrograms_db_output_np = spectrograms_db_output.cpu().numpy()

    torch.cuda.synchronize()
    print('Time to compute morphology: %.3f seconds.' % (time() - start))

    spectrograms['Closing'] = spectrogram_db_closing_np
    spectrograms['Erosion'] = spectrograms_db_erosion_np
    spectrograms['Top-hat'] = spectrograms_db_top_hat_np
    spectrograms['Output'] = spectrograms_db_output_np

    print('Time to compute morphology: %.3f' % (time() - start))

    return spectrograms


def morphology_top_hat_simplified(spectrogram_db, window, fs, n_fft, eps, device, time_resolution, frequency_precision,
                                  closing_frequency_width, closing_time_width, n_freq_bins_erosion, threshold,
                                  padding_factor, closing_time_width_final, top_hat_width):
    spectrograms = {'Input': spectrogram_db}

    # Structuring elements
    closing_shape = (int(closing_frequency_width / frequency_precision), int(closing_time_width / time_resolution))
    str_el_clo = torch.zeros(closing_shape, device=device)

    idx = np.arange(- n_freq_bins_erosion // 2 + 1, n_freq_bins_erosion // 2 + 1)
    window_spectrum = get_str_el(window, fs, n_fft, factor=padding_factor, plot=False, eps=eps).astype(np.float32)
    erosion_freq_shape = np.take(window_spectrum, idx)
    str_els_ero = torch.tensor(erosion_freq_shape, device=device, dtype=torch.float32).unsqueeze(1).unsqueeze(0)

    str_el_th = torch.zeros((top_hat_width, 1), device=device)

    str_el_clo_final = torch.zeros((1, int(closing_time_width_final / time_resolution)), device=device)

    # Computations
    start = time()
    spectrogram_db_closing = closing(spectrogram_db, str_el_clo)

    size = (1, spectrogram_db.shape[0], spectrogram_db.shape[1])
    spectrograms_db_erosion = torch.empty(size, device=device)
    spectrograms_db_top_hat = torch.empty(size, device=device)
    spectrograms_db_output = torch.zeros(size, device=device) - 1000

    for i in range(1):
        spectrograms_db_erosion[i, :, :] = erosion(spectrogram_db_closing, str_els_ero[i, :, :])
        spectrograms_db_top_hat[i, :, :] = top_hat(spectrograms_db_erosion[i, :, :], str_el_th)
        spectrograms_db_output[i, :, :][spectrograms_db_top_hat[i, :, :] > threshold] = \
            spectrograms_db_erosion[i, :, :][spectrograms_db_top_hat[i, :, :] > threshold]
        spectrograms_db_output[i, :, :] = closing(spectrograms_db_output[i, :, :], str_el_clo_final)

    spectrogram_db_closing_np = spectrogram_db_closing.cpu().numpy()
    spectrograms_db_erosion_np = spectrograms_db_erosion.cpu().numpy()
    spectrograms_db_top_hat_np = spectrograms_db_top_hat.cpu().numpy()
    spectrograms_db_output_np = spectrograms_db_output.cpu().numpy()

    torch.cuda.synchronize()
    print('Time to compute morphology: %.3f seconds.' % (time() - start))

    spectrograms['Closing'] = spectrogram_db_closing_np
    spectrograms['Erosion'] = spectrograms_db_erosion_np
    spectrograms['Top-hat'] = spectrograms_db_top_hat_np
    spectrograms['Output'] = spectrograms_db_output_np

    return spectrograms


def morphology_closing(spectrogram_db, device, time_resolution, frequency_precision,
                       closing_frequency_width, closing_time_width, opening_frequency_width, opening_time_width):
    spectrograms = {'Input': spectrogram_db}

    # Structuring elements
    closing_shape = (int(closing_frequency_width / frequency_precision), int(closing_time_width / time_resolution))
    str_el_clo = torch.zeros(closing_shape, device=device)

    if opening_time_width == 0.:
        opening_shape = (int(opening_frequency_width / frequency_precision), 1)
    else:
        opening_shape = (int(opening_frequency_width / frequency_precision), int(opening_time_width / time_resolution))
    str_el_ope = torch.zeros(opening_shape, device=device)

    # Computations
    start = time()
    spectrogram_db_closing = closing(spectrogram_db, str_el_clo)

    spectrogram_db_opening = opening(spectrogram_db_closing, str_el_ope)

    torch.cuda.synchronize()
    print('Time to compute morphology: %.3f seconds.' % (time() - start))

    spectrograms['Closing'] = spectrogram_db_closing
    spectrograms['Output'] = spectrogram_db_opening

    return spectrograms
