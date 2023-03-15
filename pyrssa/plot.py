from pyrssa.classes.SSA import SSABase
from pyrssa import Reconstruction
from pyrssa import WCorMatrix, HMatrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Literal

# This line solves problem for PyCharm: module 'backend_interagg' has no attribute 'FigureCanvas'...
matplotlib.use('TkAgg')


def _should_share_limits(series_arr, max_diff=1):
    range_values = [[min(ser), max(ser)] for ser in series_arr]
    arr_start, arr_end = list(map(list, zip(*range_values)))
    min_start = min(arr_start)
    max_end = max(arr_end)
    max_start = max(arr_start)
    min_end = min(arr_end)
    max_range = abs(min_start - max_end)
    return abs(max_start - min_start) / max_range <= max_diff and abs(max_end - min_end) / max_range <= max_diff


class Plot:

    @staticmethod
    def vectors(x: SSABase, idx=None, contrib=True, layout=None):
        if idx is None:
            idx = range(1, min(10, len(x.U)) + 1)
        if contrib is True:
            cntrb = x.contributions(idx)
        else:
            cntrb = None

        if layout is None:
            cols = 4
            rows = int(np.ceil(len(idx) / cols))
        else:
            rows = layout[0]
            cols = layout[1]

        fig = plt.figure(figsize=(cols + 2, rows + 2))
        fig.tight_layout(h_pad=1)
        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)
        fig.suptitle("Eigenvectors")
        ax = None
        for i in range(len(idx)):
            ax = fig.add_subplot(gs[i], sharey=ax)
            ax.plot(range(len(x.U[idx[i] - 1])), x.U[idx[i] - 1])
            if cntrb is None:
                ax.set_title(idx[i])
            else:
                ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f} %)')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(aspect='auto', adjustable='box')
        plt.show()

    @staticmethod
    def paired(x: SSABase, idx=None, contrib=True, layout=None):
        if idx is None:
            idx = range(len(x.U))
        if contrib is True:
            cntrb = x.contributions(idx)
        else:
            cntrb = None

        if layout is None:
            cols = 4
            rows = int(np.ceil((len(idx) - 1) / cols))
        else:
            rows = layout[0]
            cols = layout[1]

        fig = plt.figure(figsize=(cols + 2, rows + 2))
        fig.tight_layout(h_pad=1)
        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)

        fig.suptitle("Pairs of eigenvectors")
        for i in range(len(idx) - 1):
            ax = fig.add_subplot(gs[i])
            ax.plot(x.U[idx[i + 1] - 1], x.U[idx[i] - 1])
            if cntrb is None:
                ax.set_title(f'{idx[i]} vs {idx[i + 1]}')
            else:
                ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f}%) vs {idx[i + 1]} ({cntrb[i + 1] * 100:.2f}%)')
            ax.set_xticks([])
            ax.set_yticks([])
            ratio = 1.0
            x_left, x_right = ax.get_xlim()
            y_bottom, y_top = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_bottom - y_top)) * ratio)
        plt.show()

    @staticmethod
    def xyplot(x: Reconstruction, x_labels=None, add_residuals=True, add_original=True, layout=None, superpose=False):

        if not isinstance(x, Reconstruction):
            raise TypeError

        if x_labels is None:
            x_labels = x.series.index

        if superpose:
            fig, ax = plt.subplots()
            fig.tight_layout()
            if add_original:
                ax.plot(x_labels, x.series, label="Original")
            for name in x.names:
                ax.plot(x_labels, getattr(x, name), label=name)
            if add_residuals:
                ax.plot(x_labels, x.residuals, label='Residuals')
            ax.legend()
            plt.title(label="Reconstructed series")
            plt.show()

        else:

            cnt = len(x.names) + add_original + add_residuals

            if layout is not None:
                if layout[0] * layout[1] < cnt:
                    raise ValueError(f"Layout size {layout} is less than count of plotting series ({cnt}).")
                rows = layout[0]
                cols = layout[1]
            else:
                rows = cnt
                cols = 1

            plotting_series = []
            if add_original:
                plotting_series.append({'name': 'Original', 'series': x.series})
            for i in range(add_original, cnt - add_residuals):
                name = x.names[i - add_original]
                plotting_series.append({'name': name, 'series': getattr(x, name)})
            if add_residuals:
                plotting_series.append({'name': 'Residuals', 'series': x.residuals})

            if cols > 1:
                all_series = [ser['series'] for ser in plotting_series]
                share_y = _should_share_limits(all_series)
            else:
                share_y = False

            fig = plt.figure(figsize=(2 * (cols + 2), 2 * (rows + 2)))
            if share_y:
                gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0.1, hspace=0.5)
            else:
                gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, hspace=0.7)
            fig.suptitle("Reconstructed series")
            ax = None

            for i in range(len(plotting_series)):
                if share_y:
                    is_first = ax is None
                    ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
                    if not is_first:
                        ax.tick_params(left=False, labelleft=False)
                else:
                    ax = fig.add_subplot(gs[i], sharex=ax)

                ax.plot(x_labels, plotting_series[i]['series'])
                ax.set_title(plotting_series[i]['name'])

            plt.show()

    @staticmethod
    def matplot(x: Reconstruction, x_labels=None, add_residuals=True, add_original=True):

        if x_labels is None:
            x_labels = x.series.index

        fig, ax = plt.subplots()
        if add_original:
            ax.plot(x_labels, x.series, label="Original", color="black", linewidth=0.5)
        for name in x.names:
            ax.plot(x_labels, getattr(x, name), label=name, linestyle="dashed")
        if add_residuals:
            ax.plot(x_labels, x.residuals, label='Residuals', linestyle="dashed")
        ax.legend()
        plt.title(label="Reconstructed series")
        plt.show()

    @staticmethod
    def sigma(ts: SSABase):
        plt.suptitle('Component norms')
        plt.plot(ts.sigma, marker='o')
        plt.xlabel('Index')
        plt.ylabel('Norms')
        plt.yscale('log')
        plt.show()



    @staticmethod@staticmethod
    def _wcor(wcor_matrix: WCorMatrix, scales=None, support_lines=True):
        plt.imshow(wcor_matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.gca().invert_yaxis()

        if scales is None:
            ticks = range(len(wcor_matrix.groups))
            labels = wcor_matrix.groups
        else:
            ticks = np.array(scales) - 1  # fix for indexing of components on the plot
            labels = scales
            if support_lines:
                for i in scales:
                    plt.plot([i - 1, i - 1], [0, len(wcor_matrix.groups) - 1], "k-.", lw=0.5, alpha=0.5)
                    plt.plot([0, len(wcor_matrix.groups) - 1], [i - 1, i - 1], "k-.", lw=0.5, alpha=0.5)
        plt.title("W-correlation matrix")
        plt.xticks(ticks, labels=labels)
        plt.yticks(ticks, labels=labels)

        minor_cnt = wcor_matrix.shape[0]
        plt.xticks(np.arange(-.5, minor_cnt, 1), minor=True)
        plt.yticks(np.arange(-.5, minor_cnt, 1), minor=True)

        plt.grid(which='minor', color='w', linestyle='-', linewidth=0.2)
        plt.tick_params(which='minor', bottom=False, left=False)

        plt.show()
    def _hmatrix(hmatrix: HMatrix):
        plt.imshow(hmatrix.T, cmap='hot_r', origin='lower', interpolation='nearest')
        plt.title("Heterogeneity matrix")
        plt.show()

    def __call__(self, obj, x_labels=None, kind: Literal["vectors", "paired"] = None,
                 add_residuals=True, add_original=True, idx=None, scales=None,
                 contrib=True, layout=None, superpose=False, method: Literal["matplot", "xyplot"] = None,
                 support_lines=True):
        if kind == "vectors":
            return self.vectors(obj, idx=idx, contrib=contrib, layout=layout)
        elif kind == "paired":
            return self.paired(obj, idx=idx, contrib=contrib, layout=layout)
        elif type(obj) == Reconstruction:
            if method == "matplot":
                return self.matplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original)
            if method == "xyplot":
                return self.xyplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original,
                                   layout=layout, superpose=superpose)
        elif isinstance(obj, SSABase):
            return self.sigma(obj)
        elif isinstance(obj, WCorMatrix):
            return self._wcor(obj, scales=scales, support_lines=support_lines)
        elif isinstance(obj, HMatrix):
            return self._hmatrix(obj)


plot = Plot()
