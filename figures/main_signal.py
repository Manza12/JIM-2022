import numpy as np
from signal_generation import generate_harmonic, pad_and_smooth
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from pathlib import Path
from utils import create_if_not_exists, save_pickle


# Parameters
T = 4.  # in seconds
fs = 44100  # in Hertz
N = 16
f_0 = 2*110.  # in Hertz
A_0 = 0.1
delta = 1e-3

attack = 0.02  # in seconds
release = 0.05  # in seconds
init_rest = 1.  # in seconds
final_rest = 1.  # in seconds

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
}

# Harmonic
t, signal_harmonic = generate_harmonic(T, fs, N, f_0, A_0, delta)
t, signal_harmonic = pad_and_smooth(signal_harmonic, t, fs, attack, release, init_rest, final_rest)

# Save
folder = Path('..') / Path('output') / Path('synthesized')
create_if_not_exists(folder)

wav.write(folder / Path('signal_harmonic.wav'), fs, signal_harmonic)

save_pickle(folder / Path('parameters.pickle'), parameters)

# Plot
plt.figure()
plt.plot(t, signal_harmonic)
plt.show()
