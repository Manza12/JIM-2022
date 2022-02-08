import numpy as np
import scipy.signal as sig


def generate_harmonic_data(T, fs, N, f_0, A_0, delta, **kwargs):

    resulting_arrays = list()

    # Time vector
    t = np.arange(0, T, 1 / fs)

    f_n = f_0 * np.arange(1, N + 1, 1)
    omega_n = np.expand_dims(f_n, 1) * np.expand_dims(np.ones_like(t), 0)

    A_n = A_0 / np.arange(1, N + 1, 1) ** 2
    delta_n = delta * f_n
    a_n = np.expand_dims(A_n, 1) * np.exp(- 2 * np.pi * np.expand_dims(delta_n, 1) * np.expand_dims(t, 0))

    for n in range(N):
        frequencies = omega_n[n, :]
        amplitudes = smooth(a_n[n, :], t, fs, kwargs['attack'], kwargs['release'])
        timestamps = kwargs['init_rest'] + t

        resulting_array = np.array((frequencies, timestamps, amplitudes))
        resulting_arrays.append(resulting_array)

    return resulting_arrays


def generate_harmonic(T, fs, N, f_0, A_0, delta):
    # Time vector
    t = np.arange(0, T, 1 / fs)

    # Sinusoids
    f_n = f_0 * np.arange(1, N + 1, 1)
    assert f_n.max() < fs / 2, 'Nyquist frequency surpassed'
    s_n = np.sin(2 * np.pi * np.expand_dims(f_n, 1) * np.expand_dims(t, 0))

    # Amplitudes
    A_n = A_0 / np.arange(1, N + 1, 1)**2
    delta_n = delta * f_n
    a_n = np.expand_dims(A_n, 1) * np.exp(- 2 * np.pi * np.expand_dims(delta_n, 1) * np.expand_dims(t, 0))

    # Sum
    signal = np.sum(a_n * s_n, axis=0)

    return t, signal


def generate_inharmonic(T, fs, t_fft, t_hop, f_1, f_2, t_1, t_2, eta, shape='square'):
    # STFT parameters
    n_per_seg = int(t_fft * fs)
    n_overlap = int((t_fft-t_hop) * fs)

    # Time vector
    t = np.arange(0, T, 1 / fs)

    # White noise
    w = np.random.rand(len(t)) * 2 - 1

    # STFT
    omega, tau, stft = sig.stft(w, fs=fs, window='hann', nperseg=n_per_seg, noverlap=n_overlap)

    # Filter
    H = np.zeros_like(stft)
    H = create_filter(H, omega, f_1, f_2, tau, t_1, t_2, eta, shape=shape)

    # Multiplication
    stft *= H

    # iSTFT
    t, signal_inharmonic = sig.istft(stft, fs=fs, window='hann', nperseg=n_per_seg, noverlap=n_overlap)

    return t, signal_inharmonic


def create_filter(H, f, f_1, f_2, t, t_1, t_2, eta, shape='triangle'):
    if shape == 'triangle':
        t_on = np.where(np.logical_and(t_1 <= t, t < t_2))
        for idx_t in range(len(t_on[0])):
            f_high = f_2 - (f_2 - f_1) * (t[idx_t] - t_1) / (t_2 - t_1)
            H[:, idx_t] = np.logical_and(f_1 <= f, f < f_high).astype(np.float64) * np.exp(- 2 * np.pi * eta * t[idx_t])
        return H
    elif shape == 'square':
        t_on = np.where(np.logical_and(t_1 <= t, t < t_2))
        for idx_t in range(len(t_on[0])):
            H[:, idx_t] = np.logical_and(f_1 <= f, f < f_2).astype(np.float64) * np.exp(- 2 * np.pi * eta * t[idx_t])
        return H


def pad_and_smooth(signal, t, fs, attack, release, init_rest, final_rest):
    duration = len(t) / fs
    t = np.arange(-init_rest, duration + final_rest, 1 / fs)
    signal[0: int(attack * fs)] *= np.arange(0, attack, 1 / fs) / attack
    signal[-int(release * fs):] *= np.flip(np.arange(0, release, 1 / fs) / release)
    new_signal = np.concatenate((np.zeros(int(init_rest * fs)), signal, np.zeros(int(final_rest * fs))))

    return new_signal.astype(np.float32)


def pad_and_cut(signal, duration, fs, init_rest, final_rest, fade_out):
    N = int(fs * duration)
    if signal.dtype.type == np.int16:
        signal = signal / np.iinfo(signal.dtype).max
    signal = signal[0:N]
    signal[-int(fade_out * fs):] *= np.flip(np.arange(0, fade_out, 1 / fs) / fade_out)
    new_signal = np.concatenate((np.zeros(int(init_rest * fs)), signal, np.zeros(int(final_rest * fs))))

    return new_signal.astype(np.float32)


def smooth(signal, t, fs, attack, release):
    signal[0: int(attack * fs)] *= np.arange(0, attack, 1 / fs) / attack
    signal[-int(release * fs):] *= np.flip(np.arange(0, release, 1 / fs) / release)

    return signal
