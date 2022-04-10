import pickle
from signal_generation import generate_harmonic, pad_and_smooth, generate_inharmonic
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from pathlib import Path
import sys

try:
    note = sys.argv[1]

    try:
        note = int(note)
        assert note > 0, 'Note number should be a positive integer.'
    except ValueError:
        raise ValueError('Note number should be an integer.')
except IndexError:
    raise ValueError('One parameter corresponding to the midi number is needed.')

# Parameters
T = 4.  # in seconds
fs = 44100  # in Hertz
N = 16
f_0 = 440 * 2**((note - 69) / 12)  # in Hertz
A_0 = 0.1
delta = 5e-4

attack = 0.02  # in seconds
release = 0.05  # in seconds
init_rest = 0.  # in seconds
final_rest = 1.  # in seconds

L = 0.1  # in seconds
H = 0.01  # in seconds
omega_low = 100.  # in Hertz
omega_high = 300.  # in Hertz
tau_start = 0.  # in seconds
tau_end = 4.  # in seconds
eta = 1.  # in Hertz

parameters = {
    'T': T,
    'fs': fs,
    'N': N,
    'f_0': f_0,
    'A_0': A_0,
    'delta': delta,

    'attack': attack,
    'release': release,
    'init_rest': init_rest,
    'final_rest': final_rest,

    'L': L,
    'H': H,
    'omega_low': omega_low,
    'omega_high': omega_high,
    'tau_start': tau_start,
    'tau_end': tau_end,
    'eta': eta
}

# Harmonic
t_har, signal_harmonic = generate_harmonic(T, fs, N, f_0, A_0, delta)
t_inh, signal_inharmonic = generate_inharmonic(T, fs, L, H, omega_low, omega_high, tau_start, tau_end, eta)

assert len(t_har) == len(t_inh)

t_har, signal_harmonic = pad_and_smooth(signal_harmonic, t_har, fs, attack, release, init_rest, final_rest)
t_inh, signal_inharmonic = pad_and_smooth(signal_inharmonic, t_inh, fs, attack, release, init_rest, final_rest)

assert len(t_har) == len(t_inh)

signal = signal_harmonic + signal_inharmonic
t = t_har

# Save
folder = Path('..') / Path(str(note))
folder.mkdir(parents=True, exist_ok=True)

wav.write(folder / Path('synthetic_' + str(note) + '_harmonic.wav'), fs, signal_harmonic)
wav.write(folder / Path('synthetic_' + str(note) + '_inharmonic.wav'), fs, signal_inharmonic)
wav.write(folder / Path('synthetic_' + str(note) + '.wav'), fs, signal)

pickle.dump(parameters, open(folder / Path('parameters.pickle'), 'wb'))


# Plot
def plot_signal(s, x):
    fig = plt.figure()
    plt.plot(x, s)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    return fig


plot_signal(signal_harmonic, t_har)
plot_signal(signal_inharmonic, t_inh)
plot_signal(signal, t)

plt.show()

if __name__ == '__main__':
    pass
