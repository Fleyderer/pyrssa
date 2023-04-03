import pandas as pd

from rpy2 import robjects
from pyrssa.classes.SSA import SSABase
from pyrssa import Reconstruction, reconstruct
from pyrssa import WCorMatrix, HMatrix
from pyrssa import GroupPgram, GroupWCor
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Literal

# This line solves problem for PyCharm: module 'backend_interagg' has no attribute 'FigureCanvas'...
matplotlib.use('TkAgg')
_num_complete = robjects.r('utils::getFromNamespace("num.complete", "Rssa")')


def _should_share_limits(series_arr, max_diff=1, max_mul=1.5):
    range_values = [[min(ser), max(ser), max(ser) - min(ser)] for ser in series_arr]
    arr_start, arr_end, arr_range = list(map(list, zip(*range_values)))
    min_start = min(arr_start)
    max_end = max(arr_end)
    max_start = max(arr_start)
    min_end = min(arr_end)
    max_range = abs(min_start - max_end)
    return abs(max_start - min_start) / max_range <= max_diff \
        and abs(max_end - min_end) / max_range <= max_diff \
        and max_range / max(arr_range) <= max_mul


class Plot:

    @staticmethod
    def set_style(style_name="seaborn-v0_8-whitegrid"):
        plt.style.use(style_name)

    @staticmethod
    def vectors(x: SSABase, idx=None, contrib=True, layout=None, title=None):
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
        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)
        fig.suptitle("Eigenvectors" if title is None else title)
        fig.tight_layout()
        ax = None

        for i in range(len(idx)):
            ax = fig.add_subplot(gs[i], sharey=ax)
            ax.plot(x.U[idx[i] - 1])
            ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f} %)' if cntrb is not None else idx[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)

        plt.show()

    @staticmethod
    def paired(x: SSABase, idx=None, contrib=True, layout=None, title=None):
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
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)
        fig.suptitle("Pairs of eigenvectors" if title is None else title)

        for i in range(len(idx) - 1):
            ax = fig.add_subplot(gs[i])
            ax.plot(x.U[idx[i + 1] - 1], x.U[idx[i] - 1])
            if cntrb is None:
                ax.set_title(f'{idx[i]} vs {idx[i + 1]}')
            else:
                ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f}%) vs {idx[i + 1]} ({cntrb[i + 1] * 100:.2f}%)')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)

        plt.show()

    def series(self, x: SSABase, groups=None, layout=None, **kwargs):
        return self.xyplot(reconstruct(x, groups=groups),
                           add_residuals=False, add_original=False,
                           layout=layout, **kwargs)

    @staticmethod
    def xyplot(x: Reconstruction, x_labels=None, add_residuals=True, add_original=True,
               layout=None, superpose=False, title=None):

        if not isinstance(x, Reconstruction):
            raise TypeError(f"Only Reconstruction type of object is allowed in xyplot. You've tried to pass {type(x)}.")

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
            plt.title(label="Reconstructed series" if title is None else title)
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
    def matplot(x: Reconstruction, x_labels=None, add_residuals=True, add_original=True, title=None):

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
        plt.title(label="Reconstructed series" if title is None else title)
        plt.show()

    @staticmethod
    def sigma(ts: SSABase):
        plt.suptitle('Component norms')
        plt.plot(ts.sigma, marker='o')
        plt.xlabel('Index')
        plt.ylabel('Norms')
        plt.yscale('log')
        plt.show()

    @staticmethod
    def _wcor(wcor_matrix: WCorMatrix, scales=None):
        plt.imshow(wcor_matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.grid(color='k', alpha=0.2, linestyle='-', linewidth=0.3)
        if scales is None:
            ticks = range(len(wcor_matrix.groups))
            labels = wcor_matrix.groups
        else:
            ticks = np.array(scales) - 1  # fix for indexing of components on the plot
            labels = scales
        plt.title("W-correlation matrix")
        plt.xticks(ticks, labels=labels)
        plt.yticks(ticks, labels=labels)

        minor_cnt = wcor_matrix.shape[0]
        plt.xticks(np.arange(-.5, minor_cnt, 1), minor=True)
        plt.yticks(np.arange(-.5, minor_cnt, 1), minor=True)

        plt.grid(which='minor', color='w', linestyle='-', linewidth=0.2)
        plt.tick_params(which='minor', bottom=False, left=False)

        plt.show()

    @staticmethod
    def _hmatrix(hmatrix: HMatrix):
        plt.imshow(hmatrix.T, cmap='hot_r', origin='lower', interpolation='nearest')
        plt.title("Heterogeneity matrix")
        plt.show()

    @staticmethod
    def _group_pgram(x: GroupPgram, order=False, **kwargs):
        contrib = x.contributions
        if order:
            contrib = contrib.transform(np.sort)[::-1]
            contrib.reset_index(drop=True, inplace=True)
        contrib.plot(xlabel="Component", ylabel="Relative contribution", **kwargs)
        plt.show()

    @staticmethod
    def spectrum(obj, log=False, demean=False, detrend=True, ticks=None, tick_labels=None, limits=None):
        result = {}
        if isinstance(obj, Reconstruction):
            all_series = list(obj.items())
        else:
            if isinstance(obj, pd.Series):
                all_series = [(obj.name, obj)]
            else:
                all_series = [("series", obj)]

        fig, ax = plt.subplots()
        for name, series in all_series:
            if demean:
                series = matplotlib.mlab.detrend_mean(series)
            if detrend:
                series = matplotlib.mlab.detrend_linear(series)
            if log:
                series = np.log(series)
            result[name] = ax.magnitude_spectrum(series)

        if len(all_series) > 1:
            plt.title("Spectrum of series")
            plt.legend([s[0] for s in all_series])
        else:
            plt.title(f"Spectrum of {all_series[0][0]}")

        if ticks:
            ticks = [tick * 2 for tick in ticks]
            if tick_labels:
                plt.xticks(ticks, labels=tick_labels)
            else:
                plt.xticks(ticks)

        if limits is None:
            limits = (0, 0.5)
        limits = (limits[0] * 2, limits[1] * 2)
        plt.xlim(limits)
        plt.ylabel("Value")
        plt.show()
        return result

    def __call__(self, obj, x_labels=None, kind: Literal["vectors", "paired"] = None,
                 add_residuals=True, add_original=True, idx=None, scales=None,
                 contrib=True, layout=None, superpose=False, method: Literal["matplot", "xyplot"] = None,
                 title=None, groups=None, order=False, **kwargs):
        if kind == "vectors":
            return self.vectors(obj, idx=idx, contrib=contrib, layout=layout, title=title)
        elif kind == "paired":
            return self.paired(obj, idx=idx, contrib=contrib, layout=layout, title=title)
        elif kind == "series":
            return self.series(obj, groups=groups, layout=layout, **kwargs)
        elif isinstance(obj, Reconstruction):
            if method == "matplot":
                return self.matplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original,
                                    title=title)
            if method == "xyplot":
                return self.xyplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original,
                                   layout=layout, superpose=superpose, title=title)
        elif isinstance(obj, SSABase):
            return self.sigma(obj)
        elif isinstance(obj, WCorMatrix):
            return self._wcor(obj, scales=scales)
        elif isinstance(obj, HMatrix):
            return self._hmatrix(obj)
        elif isinstance(obj, GroupPgram):
            return self._group_pgram(obj, order=order, **kwargs)


def clplot(x, **kwargs):
    if isinstance(x, pd.DataFrame):
        x = x[x.columns[0]]
    if isinstance(x, pd.Series):
        x = x.values
    N = len(x)
    na_idx = np.hstack(np.argwhere(np.isnan(x))) + 1
    cr = [_num_complete(N, L=L, **{"na.idx": na_idx})[0] / (N - L + 1) * 100 for L in range(2, (N + 1) // 2 + 1)]
    plt.plot(cr)
    plt.title("Proportion of complete lag vectors")
    plt.ylabel("Percents")
    plt.xlabel("Window length, L")
    plt.show()



plot = Plot()
plot.set_style()

