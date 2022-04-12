import sys
from pathlib import Path; sys.path.insert(0, str(Path('..') / Path('..') / Path('..')))
from plot import plot_figures_paper
from pathlib import Path
import matplotlib.pyplot as plt


fig_size = (600, 300)

if __name__ == '__main__':
    plot_figures_paper(Path('.'), fig_size, save=True)
    plt.show()
