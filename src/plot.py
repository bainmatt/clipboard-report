"""
Custom plotting functions that map onto tidy statistical summaries.

TODO: make style work w faceting (use suplot args) (and add wrap: int option)
TODO: bring back get_col_names (return dict of any non-empty & unpack to self)
TODO: match_col_names check 1/ but ci/hdi/q, num but par_cols, dat list[float]
TODO: label_df/Precis.label() > refactor doctests (+ validate labs in val.py)

TODO: BPlot str args: group/level_prefix to ylabs (groups) and zlabs (levels)
TODO: BPlot list[float] args: x/ytick_range [start, end, increment] = None
TODO: BPlot int args: x/ytick_round_to = None, xticklab_rotation = 0
TODO: BPlot bool args: show_ticks=T, axis_offset=F, axis_trim=F
TODO: BPlot float args: refline
xx
TODO: doc BasePlot params (and add scatter, annotate method)

TODO: melt_df/Precis.melt (wide bool arg)
TODO: Precis.invert (unmelt/explode (to rows/columns))
TODO: ?move process_columns/get_names to Precis

TODO: Precis.prettify (ixs, Multi, round, drop non-pri met + _, GT $/date typ)
TODO: Precis.save (ext funcs take dat/mod/log/stat/fig + id/typ/dt)
TODO: to make_precis add opt arg 25/75 quants + record to self in get_names
"""

# flake8: noqa: F841  # *** temporary

import numpy as np
import pandas as pd
from typing import Any, Literal
from abc import ABC, abstractmethod

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from src.validate import validate_args
from src.colors import RGBType, get_palette, _process_rgb


class BasePlot(ABC):
    """
    Abstract class for plotting.

    Methods
    -------
    plot()
        Abstract method to be implemented by subclasses. If `precis` is not
        provided, it is computed using :func:`make_precis` based on parameters
        passed to the method.

    Attributes
    ----------
    precis : pd.DataFrame
        A precis DataFrame, either provided or computed by calling
        :func:`make_precis`.
    """
    @abstractmethod
    def plot(
        self,
        df: pd.DataFrame,
        data_col_pattern: str = "data",
        err_col_pattern: str = "hdi_|ci_",
        group_col: str | None = None,
        level_col: str | None = None,
        labels: list[str | None] = [None, None, "", ""],
        group_labels: list[str | None] | None = None,
        level_labels: list[str | None] | None = None,
        level_colors: list[RGBType] | None = get_palette(),
        facet_out_levels: bool = False,
        ax: Any | None = None,

        # self,
        # df: pd.DataFrame,
        # metric_cols: str | list[str] | None = None,
        # param_cols: str | list[str] | None = None,
        # err_type: Literal["hdi", "ci"] = "hdi",
        # err_prob: float = 0.94,

        # labels: list[str | None] = [None, None, "", ""],
        # group_labels: list[str | None] | None = None,
        # level_labels: list[str | None] | None = None,
        # level_colors: list[RGBType] | None = None,
        # ax: Any | None = None,
        # precis: pd.DataFrame | None = None
    ):
        """
        Abstract method for plotting.

        Arguments passed to the constructor of subclasses are
        specific to the plot type encapsulated by the subclass.

        The `plot` method in the superclass provides general logic for
        argument validation and processing. Therefore, the `plot` method
        in the subclass should call `super().plot` with the supplied
        arguments before implementing subclass-specific plotting logic.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing data to be summarized/plotted.

        metric_cols : str | None
            The metric to plot.

        param_cols : list[str] | None
            List of parameters for the independent axis.

        err_type: Literal["hdi", "ci"], default="hdi"
            The type of error bars to compute for the supplied metrics:
            - "hdi": Highest Density Interval
            - "ci" : Confidence Interval

        err_prob : float, default=0.94
            The probability mass of the error interval, 0 < `err_prob` < 1.

        labels : list[str | None], default=[None, None, "", ""]
            Labels for the table/plot axes and title, organized as
            ["{x}", "{y}", "{legend}", "{title}"].
            If None, the column name is used.
            If "", the label is suppressed.

        group_labels : list[str | None] | None, default=None
            Group labels for the first parameter supplied.
            If None, the default values are used.

        level_labels : list[str | None] | None, default=None
            Group labels for the second parameter, if supplied (the level).
            If None, the default values are used.

        level_colors : list[RGBType] | None, default=None
            A list of RGB colours ([0.0, 1.0] or [0, 255] scale).
            Length must be greater than or equal to the number of levels.

        group_order : Literal["asc", "desc"] | list[Any] | None, default="desc"
            How to sort the groups:
            - "desc": sort from highest to lowest mean
            - "asc" : sort from lowest to highest mean
            - None  : use order of the first parameter col (group) in `df`
            - list  : use the order of groups in the list,
                      which must contain all unique values in the group column.

        level_order : Literal["asc", "desc"] | list[Any] | None, default="desc"
            How to sort the levels. Follows the same strategy as `group_order`.

        facet_out_levels : bool, default=False
            If True, any supplied levels are plotted on parallel subplots.

        ax : Any | None
            Matplotlib axes object to plot on.

        precis : pd.DataFrame | None
            Pre-calculated precis DataFrame for metrics.
        """
        # TODO: Ensure len 1 or 2 for all/err, check all num except group/lvl

        self.df = df
        self.labels = labels

        self.data_col = self.df.filter(
            regex=data_col_pattern
        ).columns.to_list()[0]
        self.err_cols = self.df.filter(
            regex=err_col_pattern
        ).columns.to_list()[:2]

        # if group_col:
        #     group_col = df.filter(regex=group_col).columns.to_list()[0]
        # if level_col:
        #     level_col = df.filter(regex=level_col).columns.to_list()[0]

        # Ensure group and level column exist for plotting
        if group_col is None:
            self.group_col = "level_tmp"
            self.df[self.group_col] = 1
        else:
            self.group_col = group_col
        if level_col is None:
            self.level_col = "level_tmp"
            self.df[self.level_col] = 1
        else:
            self.level_col = level_col

        # Obtain (sorted) groups and subgroups (levels)
        self.groups = self.df[self.group_col].unique()
        self.levels = self.df[self.level_col].unique()

        # Check group_labels and level_labels
        if group_labels:
            if len(group_labels) != len(self.groups):
                raise ValueError(
                    f"group_labels must have "
                    f"len(groups) = {len(self.groups)}, "
                    f"got length {len(group_labels)}"
                )
            self.group_labels = group_labels
        else:
            self.group_labels = [None] * len(self.groups)

        if level_labels:
            if len(level_labels) != len(self.levels):
                raise ValueError(
                    f"level_labels must have "
                    f"len(levels) = {len(self.levels)}, "
                    f"got length {len(level_labels)}"
                )
            self.level_labels = level_labels
        else:
            self.level_labels = [None] * len(self.levels)

        # Process and check colours
        self.level_colors = _process_rgb(level_colors, scale_mode="compress")

        if len(self.level_colors) < len(self.levels):
            raise ValueError(
                f"level_colors must have "
                f"len(levels) ≥ {len(self.levels)}, "
                f"got length {len(self.level_colors)}"
            )


class RidgePlot(BasePlot):
    """
    Produce a ridge plot.

    This class is a subclass of :class:`BasePlot`, designed specifically
    to create ridge plots.

    Parameters
    ----------
    disttype : Literal["kde", "step", "combined"], default="kde"
        Specifies the type of distribution plot to use.
    plot_rug : bool, default=False
        Option to plot a rug plot beneath the distribution.
    plot_hdi : bool, default=True
        Option to plot the HDI (Highest Density Interval).
    **kwargs : dict
        Additional keyword arguments dynamically set as attributes.

    Examples
    --------
    >>> import pandas as pd
    >>> from src.precis import Precis
    >>> from src.plot import RidgePlot
    >>> from src.paths import get_path_to
    >>> from src.colors import get_palette
    >>> from src.stylesheet import customize_plots
    >>> customize_plots()

    >>> df = pd.read_parquet(
    ...     path=get_path_to("data", "raw", "flights.parquet"),
    ...     engine="pyarrow"
    ... )

    Get groups

    >>> groups = (
    ...     df.groupby("country")["total_flights"]
    ...     .mean().nlargest(5).index
    ... )
    >>> df = df[df["country"].isin(groups)]

    Get levels

    >>> df["day"] = pd.to_datetime(df["day"])
    >>> df["day_of_week"] = df["day"].dt.day_name()
    >>> levels = (
    ...     df.groupby("day_of_week")["total_flights"]
    ...     .mean().nlargest(7).index
    ... )
    >>> df = df[df["day_of_week"].isin(levels)]
    >>> print(df.head())  # doctest: +ELLIPSIS
        country        day  ...  total_flights  day_of_week
    534  Canada 2020-01-01  ...         1175.0    Wednesday
    535  Canada 2020-01-02  ...         1268.0     Thursday
    ...

    Create precis

    >>> precis = Precis(
    ...     df=df,
    ...     metric_cols="total_flights",
    ...     param_cols=[
    ...         "country",
    ...         #"day_of_week"
    ... ]
    ... ).sort(
    ...     metric_col="mean_total_flights",
    ...     sort_order="desc"
    ... )
    >>> print(precis.precis.head())  # doctest: +ELLIPSIS
                        country  ... hdi_97%_total_flights
    4  United States of America  ...               16468.0
    2                   Germany  ...                2597.0
    3            United Kingdom  ...                1688.0
    ...

    >>> fig = RidgePlot(
    ...     disttype="kde",
    ...     plot_hdi=True,
    ...     plot_rug=False
    ... ).plot(
    ...     df=precis.precis,
    ...     group_col="country",
    ...     #level_col="day_of_week",
    ...     labels=["total flights", "", "", "Daily outgoing flights"],
    ...     group_labels=["USA", None, "UK", None, None],
    ...     level_colors=get_palette("husl", 7),
    ...     facet_out_levels=True
    ... )
    >>> print(fig.df.head())  # doctest: +ELLIPSIS
                        country  ... hdi_97%_total_flights
    4  United States of America  ...               16468.0
    ...
    """
    def __init__(
        self,
        disttype: Literal["kde", "step", "combined"] = "kde",
        plot_rug: bool = False,
        plot_hdi: bool = True,
        **kwargs
    ):
        self.plot_rug = plot_rug
        self.plot_hdi = plot_hdi
        self.disttype = disttype

        ridge_kws = kwargs.pop('ridge_kws', {})
        for key, value in ridge_kws.items():
            setattr(self, key, value)

    @validate_args
    def plot(
        self,
        df: pd.DataFrame,
        data_col_pattern: str = "data",
        err_col_pattern: str = "hdi_|ci_",
        group_col: str | None = None,
        level_col: str | None = None,
        labels: list[str | None] = [None, None, "", ""],
        group_labels: list[str | None] | None = None,
        level_labels: list[str | None] | None = None,
        level_colors: list[RGBType] | None = get_palette(),
        facet_out_levels: bool = False,
        ax: Any | None = None
    ):
        super().plot(
            df=df,
            data_col_pattern=data_col_pattern,
            err_col_pattern=err_col_pattern,
            group_col=group_col,
            level_col=level_col,
            labels=labels,
            group_labels=group_labels,
            level_labels=level_labels,
            level_colors=level_colors
        )

        figwidth = 5
        figheight = max(len(self.groups) - 1, 3)
        fig, axes = plt.subplots(
            len(self.groups),
            len(self.levels) if facet_out_levels else 1,
            figsize=(figwidth, figheight),
            sharex=True,
            # sharey=True
        )
        axes = np.atleast_1d(axes)
        if axes.ndim == 1:
            axes = np.expand_dims(axes, axis=1)

        # Plot dist for each level within each group
        for i, group in enumerate(self.groups):
            group_df = self.df[self.df[self.group_col] == group]

            for j, level in enumerate(self.levels):
                ax = axes[i, j if facet_out_levels else 0]

                level_df = group_df[group_df[self.level_col] == level]
                level_values = level_df[self.data_col].tolist()
                level_hdi = [
                    level_df[self.err_cols[0]],
                    level_df[self.err_cols[1]]
                ]

                if len(self.levels) > 1:
                    level_label = (
                        self.level_labels[j]
                        if self.level_labels[j] else level
                    )
                else:
                    level_label = None

                # Plot dist
                if self.disttype == "kde":
                    sns.kdeplot(
                        level_values,
                        ax=ax,
                        label=level_label,
                        fill=True,
                        palette=[self.level_colors[j]],
                        alpha=0.5,
                        legend=False
                    )
                elif self.disttype in ("step", "combined"):
                    values = self.df.explode(self.data_col)[self.data_col]
                    group_values = (
                        group_df.explode(self.data_col)[self.data_col]
                    )
                    xmin, xmax = min(group_values), max(group_values)

                    sns.histplot(
                        level_values,
                        ax=ax,
                        bins=round((xmax - xmin) / (max(values) / 50)),
                        kde=True if self.disttype == "combined" else False,
                        element="step",
                        label=level_label,
                        fill=True,
                        palette=[self.level_colors[j]],
                        alpha=0.5,
                        legend=False,
                        # edgecolor="white"
                    )
                    # ax.hist(
                    #     level_values,
                    #     bins=round((xmax - xmin) / (max(values) / 50)),
                    #     color=self.level_colors[j],
                    #     density=True,
                    #     orientation="vertical",
                    #     histtype="stepfilled",
                    #     alpha=0.5,
                    #     edgecolor="white",
                    #     label=level_label,
                    # )
                if self.plot_rug:
                    sns.rugplot(
                        level_values,
                        ax=ax,
                        height=.095,
                        palette=[self.level_colors[j]],
                        alpha=.75,
                        legend=False
                    )
                if self.plot_hdi:
                    ax.plot(
                        [level_hdi[0], level_hdi[1]], [0, 0],
                        color=self.level_colors[j],
                        lw=2.5,
                        alpha=1
                    )

                # *************************

                # -- Group styling -------------------------------------------

                # ax.set_ylabel(
                #     self.group_labels[i] if self.group_labels[i] else group,
                #     rotation=0,
                #     labelpad=15,
                #     va="center",
                #     ha="right"
                # )

                if j == 0 or facet_out_levels:
                    if len(self.groups) > 1:
                        ax.set_ylabel("")
                        ymin, ymax = ax.get_ylim()
                        ax.set_yticks([ymin + (ymax - ymin)])
                        ax.set_yticklabels(
                            [
                                self.group_labels[i]
                                if self.group_labels[i] else group
                            ],
                            rotation=0,
                            va="top",
                            ha="left"
                        )
                    else:
                        ax.set_ylabel("Density")
                        ax.set_yticklabels([])

                    # * Optional set axis tick label rounding
                    # ax.set_xticks(xticks)
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                    # * Optional set lims / ticks (np.arange(min, first, max))

                    # xmin, xmax = ax.get_xlim()
                    # xticks = ax.get_xticks()
                    # xmin = xticks[1]
                    # xmax = xticks[-2]
                    # offset = abs(xmax - xmin) / 10
                    # ax.set_xlim(xticks[1] - offset, xticks[-2] + offset)
                    # values = self.df.explode(data_col)[data_col]
                    # xmin, xmax = min(values), max(values)
                    # xticks = np.round(
                    #     np.linspace(
                    #         xmin - abs(xmax) / 2, xmax + abs(xmax) / 2, 7
                    #     ), 0
                    # )
                    # x_min, x_max = ax.get_xlim()
                    # padding = 0.1 * (x_max - x_min)
                    # ax.set_xlim(x_min - padding, x_max + padding)

                    # * Optional trim and offset
                    # sns.despine(
                    #     ax=ax,
                    #     offset=2 if len(self.groups) == 1 else 0,
                    #     left=False,
                    #     trim=False
                    # )

                    # Spines/grid
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(.5)
                    # ax.grid(axis="x")
                    # ax.set_axisbelow(True)

                    # Reflines
                    # ax.vlines(
                    #     x=0,
                    #     ymin=ymin,
                    #     ymax=ymax,
                    #     color="black",
                    #     linewidth=.4,
                    #     linestyle="--",
                    #     alpha=0.7,
                    #     zorder=-1
                    # )

                # -- Facet styling -------------------------------------------

                # Tick length
                if i == len(self.groups) - 1:
                    ticks_j = j if facet_out_levels else -1
                    # axes[i, leg_j].tick_params(axis='x', length=1.6)  # ***

                # Legend for levels
                if i == 0 and len(self.levels) > 1:
                    leg_label = (
                        self.labels[2] or self.level_col
                        if self.labels[2] != "" else None
                    )
                    leg_j = j if facet_out_levels else 0
                    legend = axes[0, leg_j].legend(
                        title=leg_label,
                        ncol=len(self.levels),
                        loc="upper center",
                        borderpad=-1.3,
                        borderaxespad=0,
                        handlelength=0,
                        handleheight=0
                    )

                    if facet_out_levels:
                        level_colors_ix = np.array([j])
                    else:
                        level_colors_ix = np.arange(0, len(self.levels))

                    for handle, text, color in zip(
                        legend.legendHandles,
                        legend.get_texts(),
                        self.level_colors[:len(self.levels)]
                    ):
                        text.set_color(color)
                        handle.set_alpha(0)

        # Figure styling
        xlabel = self.labels[0] or "metric" if self.labels[0] != "" else None
        ylabel = (
            self.labels[1] or self.group_col if self.labels[1] != "" else None
        )
        title  = self.labels[3] if self.labels[3] != "" else None
        fig.suptitle(title, y=.985, fontsize=12)

        # fig.supylabel(ylabel)
        # fig.supxlabel(xlabel)
        # axes[len(self.groups) - 1, 0].set_xlabel(xlabel)
        # axes[0, 0].set_title(title, pad=20)

        self.df.drop(
            columns=[
                col for col in ["level_tmp", "group_tmp"]
                if col in self.df.columns
            ],
            inplace=True
        )
        plt.show()
        self.axes = axes

        return self


def main():
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(DetrendAndDeseasonalize, globs=globals())

    # -- One-off tests -------------------------------------------------------

    pass


if __name__ == "__main__":
    # from src.inspection import display
    from src.stylesheet import customize_plots
    customize_plots()

    main()
