import pandas as pd

from rpy2 import robjects
from pyrssa.classes.SSA import SSABase
from pyrssa.classes.LRR import BaseLRR
from pyrssa.classes.Periodogram import Periodogram
from pyrssa.conversion import is_list
from pyrssa.classes.Parestimate import BaseParestimate
from pyrssa import Reconstruction, reconstruct
from pyrssa import Forecast
from pyrssa import WCorMatrix, HMatrix
from pyrssa import GroupPgram
from pyrssa import calc_v
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Literal, Union

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
    def get_kwargs(kwargs: dict):
        return {key: value.pop(0) if is_list(value) else value if value else None for key, value in kwargs.items()}

    @staticmethod
    def set_style(style_name="seaborn-v0_8-whitegrid"):
        plt.style.use(style_name)

    @staticmethod
    def bold(text: str):
        return r"$\bf{" + text.replace(' ', r'\ ') + "}$"

    @staticmethod
    def vectors(x: SSABase, vectors: Literal["eigen", "factor"] = "eigen",
                idx=None, contrib=True, layout=None, title=None, show=True):

        if idx is None:
            idx = range(1, min(10, len(x.U)) + 1)
        if contrib is True:
            cntrb = x.contributions(idx)
        else:
            cntrb = None

        if vectors == "eigen":
            vec, main_title = x.U, "Eigenvectors"
        else:
            vec, main_title = calc_v(x, idx=idx), "Factor vectors"

        if layout is None:
            cols = 4
            rows = int(np.ceil(len(idx) / cols))
        else:
            rows = layout[0]
            cols = layout[1]

        fig = plt.figure(figsize=(cols + 2, rows + 2))
        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)
        fig.suptitle(main_title if title is None else title, fontweight="bold")
        fig.tight_layout()
        ax = None

        for i in range(len(idx)):
            ax = fig.add_subplot(gs[i], sharey=ax)
            ax.plot(vec[idx[i] - 1], linewidth=0.5)
            ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f} %)' if cntrb is not None else idx[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)

        if show:
            plt.show()
        return plt

    @staticmethod
    def paired(x: SSABase, vectors: Literal["eigen", "factor"] = "eigen",
               idx=None, contrib=True, layout=None, title=None, show=True):

        if idx is None:
            idx = range(len(x.U))
        if contrib is True:
            cntrb = x.contributions(idx)
        else:
            cntrb = None

        if vectors == "eigen":
            vec, main_title = x.U, "Pairs of eigenvectors"
        else:
            vec, main_title = calc_v(x, idx=idx), "Pairs of factor vectors"

        if layout is None:
            cols = 4
            rows = int(np.ceil((len(idx) - 1) / cols))
        else:
            rows = layout[0]
            cols = layout[1]

        fig = plt.figure(figsize=(cols + 2, rows + 2))
        gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)
        fig.suptitle(main_title if title is None else title, fontweight="bold")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.8) --> maybe remove after 2 month (from 07.04.23)

        for i in range(len(idx) - 1):
            ax = fig.add_subplot(gs[i])
            ax.plot(vec[idx[i + 1] - 1], vec[idx[i] - 1], linewidth=0.25)
            if cntrb is None:
                ax.set_title(f'{idx[i]} vs {idx[i + 1]}')
            else:
                ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f}%) vs {idx[i + 1]} ({cntrb[i + 1] * 100:.2f}%)')
            # lim = np.min(vec[idx[i] - 1:idx[i + 1]]), np.max(vec[idx[i] - 1:idx[i + 1]])

            xlim = np.min(vec[idx[i + 1] - 1]), np.max(vec[idx[i + 1] - 1])
            ylim = np.min(vec[idx[i] - 1]), np.max(vec[idx[i] - 1])
            x_shift = abs(xlim[1] - xlim[0]) * 0.05
            y_shift = abs(ylim[1] - ylim[0]) * 0.05
            xlim = xlim[0] - x_shift, xlim[1] + x_shift
            ylim = ylim[0] - y_shift, ylim[1] + y_shift
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_box_aspect(1)

        if show:
            plt.show()
        return plt

    def series(self, x: SSABase, groups=None, layout=None, **kwargs):
        return self.xyplot(reconstruct(x, groups=groups),
                           add_residuals=False, add_original=False,
                           layout=layout, **kwargs)

    def xyplot(self, x: Reconstruction, x_labels=None, add_residuals=True, add_original=True,
               layout=None, superpose=False, title=None, show=True, **kwargs):

        if not isinstance(x, Reconstruction):
            raise TypeError(f"Only Reconstruction type of object is allowed in xyplot. You've tried to pass {type(x)}.")

        if x_labels is None:
            x_labels = x.series.index

        kwargs.setdefault("linewidth", 0.75)

        if superpose:
            fig, ax = plt.subplots()

            if add_original:
                ax.plot(x_labels, x.series, label="Original", **self.get_kwargs(kwargs))
            for name in x.names:
                ax.plot(x_labels, getattr(x, name), label=name, **self.get_kwargs(kwargs))
            if add_residuals:
                ax.plot(x_labels, x.residuals, label='Residuals', **self.get_kwargs(kwargs))
            ax.legend()
            plt.title(label=self.bold("Reconstructed Series" if title is None else title))
            fig.tight_layout()

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
            fig.suptitle("Reconstructed Series", fontweight="bold")
            ax = None

            for i in range(len(plotting_series)):
                if share_y:
                    is_first = ax is None
                    ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
                    if not is_first:
                        ax.tick_params(left=False, labelleft=False)
                else:
                    ax = fig.add_subplot(gs[i], sharex=ax)

                ax.plot(x_labels, plotting_series[i]['series'], **self.get_kwargs(kwargs))
                if len(plotting_series) > 1:
                    ax.set_title(plotting_series[i]['name'])

        if show:
            plt.show()
        return plt

    def matplot(self, x: Reconstruction, x_labels=None, add_residuals=True,
                add_original=True, title=None, show=True, **kwargs):

        if x_labels is None:
            x_labels = x.series.index

        fig, ax = plt.subplots()
        if add_original:
            ax.plot(x_labels, x.series, label="Original", color="black", linewidth=0.5, **self.get_kwargs(kwargs))
        for name in x.names:
            ax.plot(x_labels, getattr(x, name), label=name, linestyle="dashed", **self.get_kwargs(kwargs))
        if add_residuals:
            ax.plot(x_labels, x.residuals, label='Residuals', linestyle="dashed", **self.get_kwargs(kwargs))
        ax.legend()
        plt.title(label=self.bold("Reconstructed Series" if title is None else title))
        if show:
            plt.show()
        return plt

    @staticmethod
    def sigma(ts: SSABase, show=True):
        plt.suptitle('Component norms', fontweight="bold")
        plt.plot(ts.sigma, marker='o', markersize=5)
        plt.xlabel('Index')
        plt.ylabel('Norms')
        plt.yscale('log')
        plt.locator_params(axis='y', base=10 ** 0.5)
        if show:
            plt.show()
        return plt

    def _wcor(self, wcor_matrix: WCorMatrix, scales=None, show=True):
        plt.imshow(wcor_matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.grid(color='k', alpha=0.2, linestyle='-', linewidth=0.3)
        if scales is None:
            ticks = range(len(wcor_matrix.groups))
            labels = wcor_matrix.groups
        else:
            ticks = np.array(scales) - 1  # fix for indexing of components on the plot
            labels = scales
        plt.title(self.bold("W-correlation matrix"))
        plt.xticks(ticks, labels=labels)
        plt.yticks(ticks, labels=labels)

        minor_cnt = wcor_matrix.shape[0]
        plt.xticks(np.arange(-.5, minor_cnt, 1), minor=True)
        plt.yticks(np.arange(-.5, minor_cnt, 1), minor=True)

        plt.grid(which='minor', color='w', linestyle='-', linewidth=0.2)
        plt.tick_params(which='minor', bottom=False, left=False)

        if show:
            plt.show()
        return plt

    def _hmatrix(self, hmatrix: HMatrix, show=True):
        plt.imshow(hmatrix.T, cmap='hot_r', origin='lower', interpolation='nearest')
        plt.title(self.bold("Heterogeneity matrix"))
        if show:
            plt.show()
        return plt

    def _group_pgram(self, x: GroupPgram, order=False, legend_params: dict = None, show=True, **kwargs):
        contrib = x.contributions
        if order:
            contrib = contrib.transform(np.sort)[::-1]
            contrib.reset_index(drop=True, inplace=True)
        for column in contrib:
            plt.plot(contrib[column], label=column, **self.get_kwargs(kwargs))
        if legend_params is None:
            legend_params = {}
        legend_params.setdefault("loc", "lower center")
        legend_params.setdefault("bbox_to_anchor", (0.5, 1.05))
        plt.xlabel("Component")
        plt.ylabel("Relative contribution")
        plt.legend(**legend_params)
        plt.tight_layout()
        if show:
            plt.show()
        return plt

    def pgram(self, objs: Union[Periodogram, list], ticks=None, tick_labels=None, limits=None, show=True, **kwargs):
        if is_list(objs) and all([isinstance(obj, Periodogram) for obj in objs]):
            all_series = objs
        elif isinstance(objs, Periodogram):
            all_series = [objs]
        else:
            raise TypeError(f"Only Periodogram or list of periodograms are available as plotting objects,"
                            f" not {type(objs)}")

        kwargs.setdefault("linewidth", 0.75)
        fig, ax = plt.subplots()
        for i, pgram in enumerate(all_series):
            name = pgram.series.name if pgram.series.name is not None else f"Series {i + 1}"
            ax.plot(pgram.freq, pgram.spec, label=name, **self.get_kwargs(kwargs))
        if len(all_series) > 1:
            plt.title(self.bold("Spectrum of series"))
            plt.legend()
        else:
            plt.title(self.bold(f"Spectrum of {all_series[0].series.name}"))

        if ticks:
            ticks = [tick * 2 for tick in ticks]
            if tick_labels:
                plt.xticks(ticks, labels=tick_labels)
            else:
                plt.xticks(ticks)

        if limits is None:
            limits = (0, 0.5)
        shift = abs(limits[1] - limits[0]) * 0.05
        limits = (limits[0] - shift, limits[1] + shift)
        plt.xlim(limits)
        plt.ylabel("Value")

        if show:
            plt.show()
        return plt

    def _roots(self, roots, title: str = "Roots", show=True):
        real, im = zip(*[[val.real, val.imag] for val in roots])
        fig, ax = plt.subplots()
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.add_artist(plt.Circle((0, 0), 1, fill=False, color="gray"))
        ax.scatter(real, im)
        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")
        ax.set_box_aspect(1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        plt.suptitle(self.bold(title))
        if show:
            plt.show()
        return plt

    def roots_lrr(self, x: BaseLRR, show=True):
        return self._roots(x.roots(), title="Roots of Linear Recurrence Relation", show=show)

    def roots_par(self, x: BaseParestimate, show=True):
        return self._roots(x.roots, show=show)

    def forecast(self, x: Forecast, include: int, pi: bool = True,
                 shadecols: str = None, title: str = None, show=True):
        if include is None:
            include = 12
        plt.plot(x.series[-include:], color="black", linewidth=0.5)
        if pi:
            plt.fill_between(x.mean.index, x.lower, x.upper, color=shadecols)
        plt.plot(x.mean, color="black", linewidth=0.5)
        plt.title(self.bold(title))
        if show:
            plt.show()
        return plt

    def __call__(self, obj, x_labels=None, kind: Literal["vectors", "paired"] = None,
                 add_residuals=True, add_original=True, idx=None, scales=None,
                 vectors: Literal["eigen", "factor"] = "eigen", contrib=True, layout=None, superpose=False,
                 method: Literal["matplot", "xyplot"] = "xyplot", title=None, groups=None, order=False,
                 legend_params: dict = None, ticks=None, tick_labels=None, limits=None, show=True,
                 include=None, pi=True, shadecols=None, **kwargs):
        if kind == "vectors":
            return self.vectors(obj, vectors=vectors, idx=idx, contrib=contrib, layout=layout, title=title, show=show)
        elif kind == "paired":
            return self.paired(obj, vectors=vectors, idx=idx, contrib=contrib, layout=layout, title=title, show=show)
        elif kind == "series":
            return self.series(obj, groups=groups, layout=layout, show=show, **kwargs)
        elif isinstance(obj, Reconstruction):
            if method == "matplot":
                return self.matplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original,
                                    title=title, show=show, **kwargs)
            if method == "xyplot":
                return self.xyplot(obj, x_labels=x_labels, add_residuals=add_residuals, add_original=add_original,
                                   layout=layout, superpose=superpose, title=title, show=show, **kwargs)
        elif isinstance(obj, SSABase):
            return self.sigma(obj, show=show)
        elif isinstance(obj, WCorMatrix):
            return self._wcor(obj, scales=scales, show=show)
        elif isinstance(obj, HMatrix):
            return self._hmatrix(obj, show=show)
        elif isinstance(obj, GroupPgram):
            return self._group_pgram(obj, order=order, legend_params=legend_params, show=show, **kwargs)
        elif isinstance(obj, Periodogram) or isinstance(obj, list) and all([isinstance(x, Periodogram) for x in obj]):
            return self.pgram(obj, ticks=ticks, tick_labels=tick_labels, limits=limits, show=show, **kwargs)
        elif isinstance(obj, BaseLRR):
            return self.roots_lrr(obj, show=show)
        elif isinstance(obj, BaseParestimate):
            return self.roots_par(obj, show=show)
        elif isinstance(obj, Forecast):
            return self.forecast(obj, include=include, pi=pi, shadecols=shadecols, title=title, **kwargs)


def clplot(x, show=True):
    if isinstance(x, pd.DataFrame):
        x = x[x.columns[0]]
    if isinstance(x, pd.Series):
        x = x.values
    N = len(x)
    na_idx = np.hstack(np.argwhere(np.isnan(x))) + 1
    cr = [_num_complete(N, L=L, **{"na.idx": na_idx})[0] / (N - L + 1) * 100 for L in range(2, (N + 1) // 2 + 1)]
    plt.plot(cr)
    plt.title(plot.bold("Proportion of complete lag vectors"))
    plt.ylabel("Percents")
    plt.xlabel("Window length, L")
    if show:
        plt.show()
    return plt


plot = Plot()
plot.set_style()
