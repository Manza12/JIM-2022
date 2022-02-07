import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
from datetime import time as tm
import math
import numpy as np


def plot_signal(signal, t):
    fig = plt.figure()
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    return fig


def plot_time_frequency(a, t, f, v_min=0, v_max=1, c_map='Greys',
                        fig_title=None, show=True, block=True, numpy=True,
                        full_screen=False, fig_size=(640, 480),
                        freq_type='int', resolution='cs',
                        freq_label='Frequency (Hz)', time_label='Time (s)',
                        plot_units=False, freq_names=None, dpi=120,
                        backend=matplotlib.get_backend(), interpolation='antialiased',
                        color_bar=True):
    fig = plt.figure(figsize=(fig_size[0]/dpi, fig_size[1]/dpi), dpi=dpi)

    if fig_title:
        fig.suptitle(fig_title)

    ax = fig.add_subplot(111)

    if numpy:
        a_plot = a
    else:
        a_plot = a.cpu().numpy()

    im = ax.imshow(a_plot, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max,
                   origin='lower', interpolation=interpolation)

    if color_bar:
        c_bar = fig.colorbar(im, ax=ax)
        c_bar.ax.set_title('Puissance (dB)', fontsize=8)

    plt.tight_layout()

    # Freq axis
    ax.yaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos:
                           format_freq(x, pos, f, freq_type, freq_names,
                                       plot_units=plot_units)))

    # Time axis
    ax.xaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos: format_time(x, pos, t, plot_units=plot_units, resolution=resolution)))

    # Labels
    ax.set_xlabel(time_label)
    ax.set_ylabel(freq_label)

    if full_screen:
        manager = plt.get_current_fig_manager()
        if backend == 'WXAgg':
            manager.frame.Maximize(True)
        elif backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        elif backend == 'Qt5Agg':
            manager.window.showMaximized()
        else:
            raise Exception("Backend not supported.")

    if show:
        plt.show(block=block)

    return fig


def format_freq(x, pos, f, freq_type='int', freq_names=None, plot_units=False):
    if pos:
        pass
    n = int(round(x))
    if 0 <= n < f.size:
        if plot_units:
            if freq_type == 'int':
                return str(f[n].astype(int)) + " Hz"
            elif freq_type == 'str':
                return freq_names[n]
            else:
                return ""
        else:
            if freq_type == 'int':
                return str(f[n].astype(int))
            elif freq_type == 'str':
                return freq_names[n]
            else:
                return ""
    else:
        return ""


def format_time(x, pos, t, plot_units=False, resolution='cs'):
    if pos:
        pass
    n = int(round(x))
    if resolution == 's':
        if 0 <= n < t.size:
            return str(round(t[n], 3))

    if 0 <= n < t.size:
        if plot_units:
            return str(round(t[n], 3)) + " s"
        else:
            decomposition = math.modf(round(t[n], 6))
            s = round(decomposition[1])
            hours = s // (60 * 60)
            minutes = (s - hours * 60 * 60) // 60
            seconds = s - (hours*60*60) - (minutes*60)
            if resolution == 'cs':
                return tm(second=seconds, minute=minutes, hour=hours,
                          microsecond=round(decomposition[0] * 1e6)).isoformat(
                    timespec='milliseconds')[3:-1]  # [3:]
            else:
                return tm(second=seconds, minute=minutes, hour=hours,
                          microsecond=round(decomposition[0] * 1e6)).isoformat(
                    timespec='milliseconds')[3:]  # [3:]
    else:
        return ""


def update_slider(val, figure, image, array):
    image.set_data(array[val, :, :])
    figure.canvas.draw()


def plot_slider(array, frequency_vector, time_vector, title="",
                slider_title="", c_map='hot', block=False, show=True,
                v_min=-100, v_max=0, freq_label='Frequency (Hz)',
                full_screen=True, resolution='cs',
                interpolation='antialiased'):
    figure = plt.figure()
    figure.suptitle(title)

    plt.subplots_adjust(bottom=0.25)
    ax = figure.subplots()

    image = ax.imshow(array[0, :, :], cmap=c_map, aspect='auto', interpolation=interpolation,
                      vmin=v_min, vmax=v_max, origin='lower')
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])

    # Freq axis
    ax.yaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos: format_freq(x, pos,
                                                      frequency_vector)))

    # Time axis
    ax.xaxis.set_major_formatter(
        tick.FuncFormatter(lambda x, pos: format_time(x, pos, time_vector, resolution=resolution)))

    # Labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(freq_label)

    backend = matplotlib.get_backend()
    if full_screen:
        manager = plt.get_current_fig_manager()
        if backend == 'WXAgg':
            manager.frame.Maximize(True)
        elif backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        elif backend == 'Qt5Agg':
            manager.window.showMaximized()
        else:
            raise Exception("Backend not supported.")

    if show:
        plt.ion()
        plt.show(block=block)

    slider = Slider(ax_slide, slider_title, 0, array.shape[0] - 1,
                    valinit=0, valstep=1)

    return figure, image, slider


def plot_harmonics(output):
    fig, ax = plt.subplots(1, 1)
    x_lim = [0., 1.]
    y_lim = [0., 1.]
    norm = plt.Normalize(-140, 0)
    for i in range(len(output)):
        x = output[i][1, :]
        y = output[i][0, :]
        z = 20 * np.log10(output[i][2, :])
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Greys', norm=norm)
        lc.set_array(z)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        x_lim = [min(x_lim[0], x.min()), max(x_lim[1], x.max())]
        y_lim = [min(y_lim[0], y.min()), max(y_lim[1], y.max())]
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    return fig


def plot_time_frequency_2(a_0, a_1, t, f, v_min=0, v_max=1, c_map='Greys',
                          fig_title=None, show=True, block=True, numpy=True,
                          full_screen=False, fig_size=(640, 480),
                          freq_type='int', resolution='cs',
                          freq_label='Frequency (Hz)', time_label='Time (s)',
                          plot_units=False, freq_names=None, dpi=120,
                          backend=matplotlib.get_backend(), interpolation='antialiased',
                          color_bar=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                           figsize=(fig_size[0]/dpi, fig_size[1]/dpi), dpi=dpi)
    if fig_title:
        fig.suptitle(fig_title)

    if numpy:
        a_0_plot = a_0
        a_1_plot = a_1
    else:
        a_0_plot = a_0.cpu().numpy()
        a_1_plot = a_1.cpu().numpy()

    im_0 = ax[0].imshow(a_0_plot, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max,
                        origin='lower', interpolation=interpolation)
    im_1 = ax[1].imshow(a_1_plot, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max,
                        origin='lower', interpolation=interpolation)

    if color_bar:
        c_bar_0 = fig.colorbar(im_0, ax=ax[0])
        c_bar_0.ax.set_title('Puissance (dB)', fontsize=8)

        c_bar_1 = fig.colorbar(im_1, ax=ax[1])
        c_bar_1.ax.set_title('Puissance (dB)', fontsize=8)

    plt.tight_layout()

    # Freq axis
    for i in range(2):
        ax[i].yaxis.set_major_formatter(
            tick.FuncFormatter(lambda x, pos:
                               format_freq(x, pos, f, freq_type, freq_names,
                                           plot_units=plot_units)))

    # Time axis
    for i in range(2):
        ax[i].xaxis.set_major_formatter(
            tick.FuncFormatter(lambda x, pos: format_time(x, pos, t, plot_units=plot_units, resolution=resolution)))

    # Labels
    for i in range(2):
        ax[i].set_xlabel(time_label)
        ax[i].set_ylabel(freq_label)

    if full_screen:
        manager = plt.get_current_fig_manager()
        if backend == 'WXAgg':
            manager.frame.Maximize(True)
        elif backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        elif backend == 'Qt5Agg':
            manager.window.showMaximized()
        else:
            raise Exception("Backend not supported.")

    if show:
        plt.show(block=block)

    return fig


def plot_time_frequency_top_hat(a_0, a_1, t, f, v_min=0, v_max=1, c_map='Greys',
                                fig_title=None, show=True, block=True, numpy=True,
                                full_screen=False, fig_size=(640, 480),
                                freq_type='int', resolution='cs',
                                freq_label='Frequency (Hz)', time_label='Time (s)',
                                plot_units=False, freq_names=None, dpi=120,
                                backend=matplotlib.get_backend(), interpolation='antialiased',
                                color_bar=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                           figsize=(fig_size[0]/dpi, fig_size[1]/dpi), dpi=dpi)
    if fig_title:
        fig.suptitle(fig_title)

    if numpy:
        a_0_plot = a_0
        a_1_plot = a_1
    else:
        a_0_plot = a_0.cpu().numpy()
        a_1_plot = a_1.cpu().numpy()

    im_0 = ax[0].imshow(a_0_plot, cmap=c_map, aspect='auto', vmin=0, vmax=20,
                        origin='lower', interpolation=interpolation)
    im_1 = ax[1].imshow(a_1_plot, cmap=c_map, aspect='auto', vmin=v_min, vmax=v_max,
                        origin='lower', interpolation=interpolation)

    if color_bar:
        c_bar_0 = fig.colorbar(im_0, ax=ax[0])
        c_bar_0.ax.set_title('Puissance (dB)', fontsize=8)

        c_bar_1 = fig.colorbar(im_1, ax=ax[1])
        c_bar_1.ax.set_title('Puissance (dB)', fontsize=8)

    plt.tight_layout()

    # Freq axis
    for i in range(2):
        ax[i].yaxis.set_major_formatter(
            tick.FuncFormatter(lambda x, pos:
                               format_freq(x, pos, f, freq_type, freq_names,
                                           plot_units=plot_units)))

    # Time axis
    for i in range(2):
        ax[i].xaxis.set_major_formatter(
            tick.FuncFormatter(lambda x, pos: format_time(x, pos, t, plot_units=plot_units, resolution=resolution)))

    # Labels
    for i in range(2):
        ax[i].set_xlabel(time_label)
        ax[i].set_ylabel(freq_label)

    if full_screen:
        manager = plt.get_current_fig_manager()
        if backend == 'WXAgg':
            manager.frame.Maximize(True)
        elif backend == 'TkAgg':
            manager.resize(*manager.window.maxsize())
        elif backend == 'Qt5Agg':
            manager.window.showMaximized()
        else:
            raise Exception("Backend not supported.")

    if show:
        plt.show(block=block)

    return fig


def plot_harmonics_ground_truth(output_arrays, ground_truth, indexes, step, scale='log', eps=1e-20):
    fig = plt.figure(figsize=(6., 3.))

    if indexes == 'all':
        indexes = np.arange(max(len(output_arrays), len(ground_truth)))

    for idx in indexes:
        if scale == 'log':
            amplitudes_ground = 20 * np.log10(ground_truth[idx][2, :] + eps)
            amplitudes_output = 20 * np.log10(output_arrays[idx][2, :] + eps)
        else:
            amplitudes_ground = ground_truth[idx][2, :]
            amplitudes_output = output_arrays[idx][2, :]

        plt.plot(ground_truth[idx][1, :], amplitudes_ground)
        plt.scatter(output_arrays[idx][1, ::step], amplitudes_output[::step], marker='x')

    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude (dB)')

    return fig
