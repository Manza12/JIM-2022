import numpy as np


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


def pad_and_smooth(signal, t, fs, attack, release, init_rest, final_rest):
    duration = len(t) / fs
    t = np.arange(-init_rest, duration + final_rest, 1 / fs)
    signal[0: int(attack * fs)] *= np.arange(0, attack, 1 / fs) / attack
    signal[-int(release * fs):] *= np.flip(np.arange(0, release, 1 / fs) / release)
    new_signal = np.concatenate((np.zeros(int(init_rest * fs)), signal, np.zeros(int(final_rest * fs))))

    return t, new_signal
