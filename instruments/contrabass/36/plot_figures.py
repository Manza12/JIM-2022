import sys
from pathlib import Path; sys.path.insert(0, str(Path('..') / Path('..') / Path('..')))
from plot import plot_figures
from pathlib import Path
import matplotlib.pyplot as plt


fig_size = (640, 360)

if __name__ == '__main__':
    plot_figures(Path('.'), fig_size)
    plt.show()
