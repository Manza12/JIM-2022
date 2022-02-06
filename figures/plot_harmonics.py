from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from plot import plot_harmonics

name = '69'
folder = 'output'
with open(Path('..') / Path(folder) / Path(name + '.pickle'), 'rb') as f:
    output = pickle.load(f)


plot_harmonics(output)

plt.show()

if __name__ == '__main__':
    pass
