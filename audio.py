import scipy.io.wavfile as wav
import numpy as np
from time import time
import torch


def audio_main(file_path, fs, starting_rest, duration, final_fade, ts, device):
    # Read audio file
    sr, data = wav.read(file_path)
    assert fs == sr

    # Pad and clip
    signal = np.concatenate((np.zeros(int(fs * starting_rest)), data))[:int(fs * duration)].astype(np.float32)
    signal[-int(fs * final_fade):] *= np.flip(np.linspace(0, 1, int(fs * final_fade)))

    # Create time vector
    N = len(signal)
    duration = N / fs
    t = np.arange(0, duration, ts)

    # Send data to the GPU
    start = time()
    signal_tensor = torch.from_numpy(signal).to(device)
    print('Time to send signal to GPU: %.3f' % (time() - start))

    return signal, signal_tensor, t
