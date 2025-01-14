"""
Versatile statistical summary functions with plotting integration.
"""

import re
import numpy as np
import arviz as az
import pandas as pd
from typing import Any, Literal

from src.plot import RidgePlot
from src.colors import RGBType, get_palette
from src.validate import validate_args, validate_column_list


class Precis:
    """
    Wrapper class for creating precis summaries and plotting.

    This class encapsulates functionality in :func:`make_precis` and
    subclasses of :class:`BasePlot`.

    Attributes
    ----------
    precis : pd.DataFrame
        A precis computed by calling :func:`precis.make_precis`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.precis import Precis
    >>> from src.stylesheet import customize_plots
    >>> customize_plots()
    >>> np.random.seed(0)

    Make data

    >>> df = pd.DataFrame({
    ...     'fold': np.random.choice([1, 2, 3, 4, 5], 200),
    ...     'learning_rate': np.random.choice([0.01, 0.05], 200),
    ...     'num_trees': np.random.choice([50, 100, 150, 200], 200),
    ...     'accuracy': np.random.uniform(0.75, 0.95, 200),
    ...     'loss': np.random.uniform(0.1, 10, 200),
    ...     'precision': np.random.uniform(0.7, 0.95, 200),
    ...     'recall': np.random.uniform(0.7, 0.95, 200),
    ... })
    >>> mask = df["learning_rate"] == 0.05
    >>> df.loc[mask, "loss"] = df.loc[mask, "loss"] + 7

    Construct precis

    >>> precis = Precis(
    ...     df=df,
    ...     metric_cols=["loss"],
    ...     param_cols=["num_trees", "learning_rate"],
    ...     err_type="hdi",
    ...     labels=[None, r"$n_{trees} (\\epsilon)$", "", "Ridge plot"],
    ...     #group_labels=[
    ...     #    r"$group_a$", r"$group_b$", r"$group_c$", r"$group_d$"
    ...     #],
    ...     #level_labels=[r"$lvl_1 = \\mu_1$", r"$lvl_2 = \\mu_2$"],
    ... ).sort(
    ...     custom_order=dict(
    ...         num_trees=[50, 100, 150, 200],
    ...         learning_rate=[0.01, 0.05]
    ...     )
    ... )
    >>> print(precis.precis)  # doctest: +ELLIPSIS
       num_trees  learning_rate  ... hdi_3%_loss  hdi_97%_loss
    0         50           0.01  ...    0.393385      9.120852
    1         50           0.05  ...    7.111955     16.371034
    2        100           0.01  ...    0.525466      8.046652
    ...

    Plot

    >>> precis.plot(
    ...     kind="ridge",
    ...     level_colors=get_palette("husl", 2),
    ...     facet_out_levels=False,
    ...     ridge_kws=dict(disttype="kde", plot_rug=True, plot_hdi=False)
    ... )
    """
    @validate_args
    def __init__(
        self,
        df: pd.DataFrame,
        metric_cols: str | list[str] | None = None,
        param_cols: str | list[str] | None = None,
        err_type: Literal["hdi", "ci"] = "hdi",
        err_prob: float = 0.94,

        labels: list[str | None] = [None, None, "", ""],
        group_labels: list[str | None] | None = None,
        level_labels: list[str | None] | None = None,
    ):
        self.df = df
        self.err_type = err_type
        self.err_prob = err_prob

        self.metric_cols, self.param_cols = _process_column_lists(
            df=df,
            metric_cols=metric_cols,
            param_cols=param_cols
        )
        # (self.metric_col, self.group_col, self.level_col,
        #  self.data_col, self.err_cols) = _get_column_names(
        #     metric_cols=self.metric_cols,
        #     param_cols=self.param_cols,
        #     err_type=self.err_type,
        #     err_prob=self.err_prob
        # )

        self.precis = make_precis(
            df=self.df,
            metric_cols=self.metric_cols,
            param_cols=self.param_cols,
            err_type=self.err_type,
            err_prob=self.err_prob
        )

        self.labels = labels
        self.group_labels = group_labels
        self.level_labels = level_labels

    def sort(
        self,
        metric_col: str | None = None,
        sort_order:
            Literal["asc", "desc"] | list[Literal["asc", "desc"]] = "desc",
        custom_order: dict[str, list[Any]] | None = None,
    ):
        """
        Sort precis parameter cols using metric, lexical, or custom orders.

        Wrapper for :func:`precis.sort_df`.

        Returns
        -------
        self
            The Precis object with the transformed precis DataFrame attribute,
            sorted by the specified ordering.
            The output DataFrame has the same dimensions and structure as the
            input, with only the order of the rows changed.
        """
        self.precis = sort_df(
            df=self.precis,
            param_cols=self.param_cols,
            metric_col=metric_col,
            sort_order=sort_order,
            custom_order=custom_order
        )
        return self

    def plot(
        self,
        kind: Literal["ridge"],
        level_colors: list[RGBType] | None = get_palette(),
        facet_out_levels: bool = False,
        ax: Any | None = None,
        **ridge_kws
    ):
        """
        Call the plot method of the passed :class:`BasePlot` instance.

        Parameters
        ----------
        kind : Literal["ridge"]
            The kind of plot to generate.

        level_colors : list[RGBType] | None, default=None
            A list of RGB colours ([0.0, 1.0] or [0, 255] scale).
            Length must be greater than or equal to the number of levels.

        ax : Any | None
            Matplotlib axes object to plot on.
        """
        self.kind = kind
        self.level_colors = level_colors
        self.ax = ax

        if kind == "ridge":
            self.plot_function = RidgePlot(**ridge_kws)

        self.plot_function.plot(
            df=self.precis,
            group_col=self.param_cols[0],
            level_col=(
                self.param_cols[1] if len(self.param_cols) > 1 else None
            ),
            labels=self.labels,
            group_labels=self.group_labels,
            level_labels=self.level_labels,
            level_colors=self.level_colors,
            facet_out_levels=facet_out_levels,
            ax=self.ax
        )


@validate_args
def make_precis(
    df: pd.DataFrame,
    metric_cols: str | list[str] | None = None,
    param_cols: str | list[str] | None = None,
    err_type: Literal["hdi", "ci"] = "hdi",
    err_prob: float = 0.94
) -> pd.DataFrame:
    """
    Generate a precis (brief statistical summary) for numeric columns in a DataFrame.

    Hyperparameters/groups are output as rows and metrics as columns. The
    function collapses over groups such as categories or cross-validation
    folds.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric data or results of model
        evaluation (e.g., cross-validation, grid search).

    metric_cols : str | list[str]
        List of column names representing performance metrics (e.g., accuracy,
        RMSE).

    param_cols : str | list[str]
        List of column names representing hyperparameters or other identifying
        information (such as parameters or models).
        These will appear in the rows of the output.

        Alternatively, the names of columns representing groups (primary keys)
        and optionally subgroups (levels or secondary keys).
        Summary statistics are calculated over the groups for each unique set
        of parameters.

    err_type: Literal["hdi", "ci"], default="hdi"
        The type of error bars to compute for the supplied metrics:
        - "hdi": Highest Density Interval
        - "ci" : Confidence Interval

    err_prob : float, default=0.94
        The probability mass of the error interval, 0 < `err_prob` < 1.

    Returns
    -------
    precis_df : pd.DataFrame
        DataFrame with unique combinations of hyperparameters/groups as rows
        and summary statistics for metrics as columns, including:
        - mean_{metric}
        - std_{metric}
        - {err_type}_{int((1 - err_prob) / 2 * 100)}%_{metric}
        - {err_type}_{100 - int((1 - err_prob) / 2 * 100)}%_{metric}

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.precis import make_precis

    >>> pd.reset_option("display.max_columns")
    >>> pd.reset_option("display.max_colwidth")
    >>> np.random.seed(0)

    Example 1: Cross-validation results with hyperparameters

    >>> cv_results = pd.DataFrame({
    ...     "fold": np.repeat([1, 2, 3, 4, 5], 100),
    ...     "accuracy": np.random.uniform(0.7, 0.95, 500),
    ...     "rmse": np.random.uniform(0.1, 0.3, 500),
    ...     "hyperparameter_1": np.random.choice([0.1, 0.01, 0.001], 500),
    ...     "hyperparameter_2": np.random.choice([50, 100], 500)
    ... })
    >>> cv_precis = make_precis(
    ...     cv_results,
    ...     metric_cols=["accuracy", "rmse"],
    ...     param_cols=["hyperparameter_1", "hyperparameter_2"]
    ... )

    >>> print(cv_results)  # doctest: +ELLIPSIS
         fold  accuracy      rmse  hyperparameter_1  hyperparameter_2
    0       1  0.837203  0.162076             0.010               100
    1       1  0.878797  0.174607             0.010               100
    ...
    >>> with pd.option_context("display.max_columns", 3):
    ...     print(cv_precis)  # doctest: +ELLIPSIS
       hyperparameter_1  ...  hdi_97%_rmse
    0             0.001  ...      0.277959
    1             0.001  ...      0.293229
    ...

    Example 2: Random search results with 10 iterations

    >>> grid_search_results = pd.DataFrame({
    ...     # "iteration": np.repeat(range(10), 5),
    ...     "accuracy": np.random.uniform(0.7, 0.95, 50),
    ...     "rmse": np.random.uniform(0.1, 0.3, 50),
    ...     "learning_rate": np.random.choice([0.01, 0.001, 0.0001], 50),
    ...     "batch_size": np.random.choice([32, 64, 128], 50)
    ... })
    >>> grid_search_precis = make_precis(
    ...     grid_search_results,
    ...     # metric_cols=["accuracy", "rmse"],
    ...     param_cols=["learning_rate", "batch_size"]
    ... )

    >>> print(grid_search_results)  # doctest: +ELLIPSIS
        accuracy      rmse  learning_rate  batch_size
    0   0.785863  0.289127         0.0010          32
    1   0.832328  0.192351         0.0001          64
    ...
    >>> print(grid_search_precis)  # doctest: +ELLIPSIS
       learning_rate  batch_size  ... hdi_3%_rmse  hdi_97%_rmse
    0         0.0001          32  ...    0.148506      0.290132
    1         0.0001          64  ...    0.104334      0.289038
    ...

    Example 3: Learning curve data for different training sizes

    >>> learning_curve_results = pd.DataFrame({
    ...     "iteration": np.repeat(range(6), 5),
    ...     "training_size": np.repeat([100, 200, 300, 400, 500], 6),
    ...     "accuracy": np.random.uniform(0.7, 0.95, 30),
    ...     "loss": np.random.uniform(0.1, 0.4, 30),
    ...     "model": np.tile(["model a", "model b"], 15)
    ... })
    >>> learning_curve_precis = make_precis(
    ...     learning_curve_results,
    ...     metric_cols="accuracy",
    ...     param_cols=["model", "training_size"]
    ... )

    >>> print(learning_curve_results)  # doctest: +ELLIPSIS
        iteration  training_size  accuracy      loss    model
    0           0            100  0.921783  0.298473  model a
    1           0            100  0.741386  0.117707  model b
    2           0            100  0.866490  0.343295  model a
    ...
    >>> with pd.option_context("display.max_columns", 3):
    ...     print(learning_curve_precis)  # doctest: +ELLIPSIS
         model  ...  hdi_97%_accuracy
    0  model a  ...          0.943473
    1  model a  ...          0.910454
    2  model a  ...          0.889863
    ...

    Example 4: Resampling results for obtaining weights distribution

    >>> resampling_results = pd.DataFrame({
    ...     "resample_iteration": np.arange(1, 101),
    ...     "coefficient_1": np.random.normal(1, 0.1, 100),
    ...     "coefficient_2": np.random.normal(2, 0.2, 100),
    ...     "r2_score": np.random.uniform(0.6, 0.9, 100)
    ... }).round(3)
    >>> resampling_precis = make_precis(
    ...     resampling_results,
    ...     metric_cols=["coefficient_1", "coefficient_2", "r2_score"],
    ...     param_cols=[]
    ... )

    >>> print(resampling_results)  # doctest: +ELLIPSIS
        resample_iteration  coefficient_1  coefficient_2  r2_score
    0                    1          0.906          2.033     0.846
    1                    2          0.957          2.101     0.747
    2                    3          1.070          1.989     0.640
    ...
    >>> with pd.option_context(
    ...     "display.max_columns", 3,
    ...     "display.max_colwidth", 30
    ... ):
    ...     print(resampling_precis)  # doctest: +ELLIPSIS
                  data_coefficient_1  ...  hdi_97%_r2_score
    0  [0.906, 0.957, 1.07, 1.082...  ...             0.892
    ...

    Example 5: Feature selection via LOO CV

    >>> feature_selection_results = pd.DataFrame({
    ...     "loo_iteration": np.repeat(range(10), 3),
    ...     "accuracy": np.random.uniform(0.7, 0.95, 30),
    ...     "f1_score": np.random.uniform(0.6, 0.9, 30),
    ...     "feature_set": np.tile(["set a", "set b", "set c"], 10)
    ... })
    >>> feature_selection_precis = make_precis(
    ...     feature_selection_results,
    ...     metric_cols=["accuracy", "f1_score"],
    ...     param_cols="feature_set"
    ... )
    >>> print(feature_selection_results)  # doctest: +ELLIPSIS
        loo_iteration  accuracy  f1_score feature_set
    0               0  0.914884  0.772690       set a
    1               0  0.928913  0.784310       set b
    2               0  0.704808  0.620357       set c
    ...
    >>> print(feature_selection_precis)  # doctest: +ELLIPSIS
      feature_set  ... hdi_97%_f1_score
    0       set a  ...         0.888607
    1       set b  ...         0.885423
    2       set c  ...         0.889478
    ...
    """
    metric_cols, param_cols = _process_column_lists(
        df=df,
        metric_cols=metric_cols,
        param_cols=param_cols
    )

    # Group by the parameter columns (create a group object)
    if param_cols:
        grouped = df.groupby(param_cols)
    else:
        grouped = df.groupby(lambda x: True)

    # -- Loop through all parameter groups -----------------------------------

    summary_list: list[dict[str, Any]] = []

    for params, group_data in grouped:
        summary_dict: dict[str, Any] = {}

        # Make sure param_cols is iterable (can be zipped)
        if len(param_cols) == 1:
            params = (params,)
        if param_cols:
            summary_dict = {col: val for col, val in zip(param_cols, params)}

        # Compute summary stats for all metrics for all parameter groups
        for col in metric_cols:
            if col in param_cols:
                continue

            values = group_data[col].dropna().values
            values = np.asarray(values)
            if values.size == 0:
                continue

            # Compute summary stats and store dynamically using metric name
            mean_val = np.nanmean(values)
            std_val  = np.nanstd(values)
            raw_vals = list(values)
            n_vals   = len(values)

            hdi_bounds = az.hdi(values, err_prob=err_prob)

            err_lower_prob = int((1 - err_prob) / 2 * 100)
            err_lower_text = f"{err_type}_{err_lower_prob}%_{col}"
            err_upper_text = f"{err_type}_{100 - err_lower_prob}%_{col}"

            summary_dict[f"data_{col}"]  = raw_vals
            summary_dict[f"size_{col}"]  = n_vals
            summary_dict[f"mean_{col}"]  = mean_val
            summary_dict[f"std_{col}"]   = std_val
            summary_dict[err_lower_text] = hdi_bounds[0]
            summary_dict[err_upper_text] = hdi_bounds[1]

        summary_list.append(summary_dict)

    precis_df = pd.DataFrame(summary_list)
    return precis_df


@validate_args
def sort_df(
    df: pd.DataFrame,
    param_cols: list[str],
    metric_col: str | None = None,
    sort_order:
        Literal["asc", "desc"] | list[Literal["asc", "desc"]] = "desc",
    custom_order: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """
    Sort a precis by given parameters using metric, lexical, or custom orders.

    If `metric_col` is given, its mean collapsed across successive `param_cols`
    is used for hierarchical sorting.

    Parameters
    ----------
    df : pd.DataFrame
        The precis dataframe to be sorted.
    param_cols : list[str]
        Columns to sort by in hierarchical order.
    metric_col : str, default=None
        Column containing metric values for sorting within each param level.

        If provided, `param_cols` are sorted according to `sort_order`, based
        on values in the hierarchically grouped `metric_col` mean.

        If not provided, `param_cols` are sorted according to the standard
        numeric/lexical order following `sort_order`, based on the values
        in the columns themselves.
    sort_order : Literal["asc", "desc"] | list[Literal["asc", "desc"]],
                 default="desc"
        Sorting direction for each parameter column.
        If a single string, it applies to all columns.
    custom_order : dict of lists, optional
        Custom orders for columns in `param_cols`.
        Keys should be column names and values should be ordered lists of
        values in that column.

        Takes precedence over `metric_col`.

    Returns
    -------
    pd.DataFrame
        The sorted precis DataFrame. Has the same dimensions and structure as
        the input, with only the order of the rows changed.

    Raises
    ------
    ValueError
        If:
        - any column in `param_cols` or `metric_col` is missing from `df`
        - any category in `custom_order` is missing from the
          corresponding field of `df`
        - any `sort_order` is a list with length not matching that
          of `param_cols`

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from src.precis import sort_df

    Sort by metric means collapsed across successive parameter columns

    >>> df = pd.DataFrame({
    ...     "group": np.repeat(["a", "b", "c"], 2),
    ...     "level": np.tile(["i", "ii"], 3),
    ...     "metric": [2, 1, 4, 3, 6, 5]
    ... })
    >>> sort_df(
    ...     df,
    ...     param_cols=["group", "level"],
    ...     metric_col="metric",
    ...     sort_order=["desc", "desc"]
    ... )
      group level  metric
    4     c     i       6
    5     c    ii       5
    2     b     i       4
    3     b    ii       3
    0     a     i       2
    1     a    ii       1

    Sort by lexical orders

    >>> sort_df(
    ...     df,
    ...     param_cols=["group", "level"],
    ...     sort_order=["asc", "desc"]
    ... )
      group level  metric
    1     a    ii       1
    0     a     i       2
    3     b    ii       3
    2     b     i       4
    5     c    ii       5
    4     c     i       6

    Sort by custom orders

    >>> sort_df(
    ...     df,
    ...     param_cols=["group", "level"],
    ...     custom_order={"group": ["b", "a", "c"], "level": ["i", "ii"]}
    ... )
      group level  metric
    2     b     i       4
    3     b    ii       3
    0     a     i       2
    1     a    ii       1
    4     c     i       6
    5     c    ii       5

    Sort by a mixture of lexical and custom order

    >>> sort_df(
    ...     df,
    ...     param_cols=["group", "level"],
    ...     sort_order="desc",
    ...     custom_order={"group": ["b", "a", "c"]}
    ... )
      group level  metric
    3     b    ii       3
    2     b     i       4
    1     a    ii       1
    0     a     i       2
    5     c    ii       5
    4     c     i       6
    """
    # Ensure `sort_order` is a list matching `param_cols` length
    if isinstance(sort_order, str):
        sort_order = [sort_order] * len(param_cols)
    elif len(sort_order) != len(param_cols):
        raise ValueError(
            f"length of `sort_order` ({len(sort_order)})"
            f"does not match length of `param_cols` ({len(param_cols)})."
        )
    if custom_order is None:
        custom_order = dict()
    else:
        validate_column_list(
            list(custom_order.keys()), "custom_order", df=df
        )

    # Collect columns to sort on
    order_cols: list[str] = []
    ascending_flags: list[bool] = []

    for i, col in enumerate(param_cols):
        # Sort the param col by the custom order if given
        if col in custom_order:
            missing_cats = set(df[col].unique()) - set(custom_order[col])
            if missing_cats:
                raise ValueError(
                    f"{custom_order[col]} is missing the following values: "
                    f"{sorted(missing_cats)}"
                )
            df[f"order_{i}"] = df[col].map(
                {key: ix for ix, key in enumerate(custom_order[col])}
            )
            # Order should match custom_order
            order_cols.append(f"order_{i}")
            ascending_flags.append(True)

        # Sort by a partitioned mean of metric_col within the param level
        elif metric_col:
            df[f"order_{i}"] = (
                df.groupby(col)[metric_col].transform("mean")
            )
            order_cols.append(f"order_{i}")
            ascending_flags.append(sort_order[i] == "asc")

        # Sort directly by param col if no custom order or metric_col provided
        else:
            order_cols.append(col)
            ascending_flags.append(sort_order[i] == "asc")

    # Sort and drop temporary order columns
    df = df.sort_values(by=order_cols, ascending=ascending_flags)
    df = df.drop(columns=df.filter(like='order_').columns.tolist())

    return df


def _process_column_lists(
    df: pd.DataFrame,
    metric_cols: str | list[str] | None = None,
    param_cols: str | list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Private input processing function called by precis and plot routines.

    Ensure metric_cols and param_cols are formatted as lists of strings.
    """
    if metric_cols is None:
        metric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if param_cols is None:
        param_cols = []

    if isinstance(metric_cols, str):
        metric_cols = [metric_cols]
    if isinstance(param_cols, str):
        param_cols = [param_cols]

    return metric_cols, param_cols


def _get_column_names(
    metric_cols: list[str],
    param_cols: list[str],
    err_type: Literal["hdi", "ci"] = "hdi",
    err_prob: float = 0.94
) -> tuple[str, str, str | None, str, list[str]]:
    """Private input processing function called by precis and plot routines.

    Obtain the names of the primary metric column, group and level column,
    and data and error columns.
    """
    metric_col = str(metric_cols[0])
    data_col = f"data_{metric_col}"

    err_lower_prob = int((1 - err_prob) / 2 * 100)
    err_lower_text = f"{err_type}_{err_lower_prob}%_{metric_col}"
    err_upper_text = f"{err_type}_{100 - err_lower_prob}%_{metric_col}"
    err_cols = [err_lower_text, err_upper_text]

    group_col = param_cols[0]
    if len(param_cols) > 1:
        level_col = param_cols[1]
    else:
        level_col = None

    return metric_col, group_col, level_col, data_col, err_cols


def main():
    import doctest
    doctest.testmod(verbose=True)

    # from src.workflow import doctest_function
    # doctest_function(sort_df, globs=globals())

    # -- One-off tests -------------------------------------------------------

    pass


if __name__ == "__main__":
    from src.inspection import display
    from src.stylesheet import customize_plots
    customize_plots()

    with pd.option_context('display.max_columns', None):
        # print(grid_search_precis)
        # display("grid_search_precis", globs=globals())
        pass

    main()

    # exit()

    ## Ridgeplot tests

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    n_groups = 3
    n_levels = 3
    samples_per_level = 100
    variance = 25
    sep = 30

    groups = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    levels = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

    # Create equal-sized subgroups of increasing magnitude
    df = pd.DataFrame({
        'value': np.concatenate([
            np.random.uniform(start, start + variance, samples_per_level)
            for start in range(0, (n_groups * n_levels) * sep, sep)
        ]),
        'group': np.repeat(
            groups[:n_groups], samples_per_level * n_levels
        ),
        'level': np.tile(
            np.repeat(levels[:n_levels], samples_per_level),
            n_groups
        )
    })
    level_colors = get_palette("husl", n_levels)
    g = sns.FacetGrid(
        df,
        row="group",
        hue="level",
        aspect=5,
        height=1,
        # legend_out=True,
        palette=level_colors
    )
    # Draw the densities
    g.map(
        sns.kdeplot,
        "value",
        fill=True,
        clip_on=False,
        alpha=0.5,
        # lw=1.5,
        # bw_adjust=0.5
    )
    g.add_legend(
        title="",
        ncol=len(df['level']),
        loc='upper left',
        # borderpad=-.5,
        # borderaxespad=1,
        handlelength=0,
        handleheight=0,
    )
    legend = g._legend
    for handle, text, color in zip(
        legend.legendHandles,
        legend.get_texts(),
        level_colors[:len(df['level'])]
    ):
        text.set_color(color)
        handle.set_alpha(0)

    g.set(yticks=[], ylabel="")
    for ax in g.axes.flat:
        # ax.tick_params(axis='x', length=2, rotation=0)
        # ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
        pass
    g.despine(left=True, trim=False)
    plt.show()

    ## Summarize and invert

    resampling_results = pd.DataFrame({
        "resample_Iteration": np.arange(1, 101),
        "group": np.random.choice(["group_a", "group_b", "group_c"], 100),
        "level": np.random.choice(["level_a", "level_b"], 100),
        "coefficient_1": np.random.normal(1, 0.1, 100),
        "coefficient_2": np.random.normal(2, 0.2, 100),
        "parameter_1": np.random.normal(2, 0.2, 100),
        "parameter_2": np.random.normal(2, 0.2, 100),
        "r2_score": np.random.uniform(0.6, 0.9, 100)
    }).round(3)
    with pd.option_context('display.max_columns', None,):
        display("resampling_results", globs=globals())

    resampling_precis = make_precis(
        resampling_results,
        metric_cols=[
            "coefficient_1", "coefficient_2",
            "parameter_1", "parameter_2",
        ],
        param_cols=["group", "level"]
    ).set_index(["group", "level"])
    with pd.option_context('display.max_columns', None,):
        display("resampling_precis", globs=globals())

    # Invert summary
    inverse_resampling_precis = resampling_precis.reset_index().explode(
        column=list(resampling_precis.filter(regex="^data_").columns)
    )
    with pd.option_context('display.max_columns', None,):
        display("inverse_resampling_precis", globs=globals())

    ## Melt and invert

    # Melt data for various learned params (e.g. MC chains/resampling results)
    # into rows (wide_to_long transformation).
    # TODO: use as suffixes the metric_cols (if only one drop underscores)
    id_cols = ["group", "level"]
    stub_patterns   = r"(_coefficient_\d+|_parameter_\d+)"
    suffix_patterns = r"(coefficient_\d+|parameter_\d+)"

    stubnames = [
        col for col in resampling_precis.columns
        .str.replace(stub_patterns, '', regex=True).unique()
        if col not in id_cols
    ]

    # Verify no missing columns between stubs and ids
    stub_cols = [
        col for col in resampling_precis.columns
        if re.search(stub_patterns, col)
    ]
    missing_cols = (
        set(resampling_precis.columns)
        .difference(set(id_cols + stub_cols))
    )
    msg = (
        "id_cols and stubnames must account for all columns in precis, "
        "the following columns are missing: "
        f"{sorted(missing_cols)}"
    )
    assert len(missing_cols) == 0, msg

    # Melt
    melted_resampling_precis = pd.wide_to_long(
        df=resampling_precis.reset_index(),
        stubnames=stubnames,
        i=id_cols,
        j="params",
        sep="_",
        suffix=suffix_patterns
    )
    with pd.option_context('display.max_columns', None):
        display("melted_resampling_precis", globs=globals())

    # Invert melt
    inverse_melted_resampling_precis = melted_resampling_precis.unstack()
    inverse_melted_resampling_precis.columns = (
        inverse_melted_resampling_precis.columns
        .map('{0[0]}_{0[1]}'.format)
    )
    with pd.option_context('display.max_columns', None):
        display("inverse_melted_resampling_precis", globs=globals())

    from great_tables import GT
    (
        GT(melted_resampling_precis)
        .fmt_scientific(columns="Mean")
    )

    ## Create multi-index for display

    # Group columns based on their prefixes
    data_cols = (
        resampling_precis.filter(like='data_').columns.tolist()
    )
    size_cols = (
        resampling_precis.filter(like='size_').columns.tolist()
    )
    mean_cols = (
        resampling_precis.filter(like='mean_').columns.tolist()
    )
    stddev_cols = (
        resampling_precis.filter(like='std_').columns.tolist()
    )
    err_cols = (
        resampling_precis.filter(like='hdi_').columns.tolist()
    )
    # Create a MultiIndex for the columns
    multi_cols = pd.MultiIndex.from_tuples(
        # [("Group", col) for col in id_cols] +
        [("data", col) for col in data_cols] +
        [("size", col) for col in size_cols] +
        [("mean", col) for col in mean_cols] +
        [("std", col) for col in stddev_cols] +
        [("error", col) for col in err_cols]
    )

    # Assign the MultiIndex to the DataFrame
    multi_resampling_precis = resampling_precis.copy()
    multi_resampling_precis.columns = multi_cols
    with pd.option_context('display.max_columns', None):
        display("multi_resampling_precis", globs=globals())

    ## Forest plot tests

    # Example data
    np.random.seed(42)
    groups = [
        'a',
        'group_b',
        'group_c',
        'group_d',
        'group_e',
        'group_f',
        'group_g',
        'group_h',
        'group_i',
        # 'group_j',
    ]
    levels = [
        # '',
        'level_1',
        'level_2',
        # 'level_3',
        # 'level_4',
    ]

    data = []
    for group in groups:
        for level in levels:
            mean = np.random.uniform(0.5, 1.4)
            hdil = np.random.uniform(0.1, 0.5)
            hdiu = np.random.uniform(1.5, 1.8)
            q25 = np.random.uniform(0.3, 0.5)
            q75 = np.random.uniform(1.4, 1.5)
            data.append([group, level, mean, hdil, hdiu, q25, q75])

    melted_resampling_precis = pd.DataFrame(
        data, columns=['group', 'level', 'mean', 'hdil', 'hdiu', 'q25', 'q75']
    )

    # Plot
    levels_melted = melted_resampling_precis['level'].unique()
    groups_melted = melted_resampling_precis['group'].unique()

    # -- Compute dimensions and group widths/centers -------------------------

    # *Distance between levels within each group
    # and space between neighbouring levels of different groups.
    level_offset = 1.0 / 2
    group_spacing = 1.0 if len(levels_melted) > 1 else level_offset

    # Width occupied by each group and distance between group centers
    group_width = (
        (len(levels_melted) - 1) * level_offset
        if len(levels_melted) > 1 else 0
    )
    group_offset = group_width + group_spacing

    # Group centers (tick positions) with initial 1/2 group width factored in
    group_centers = [
        (group_width / 2) + i * group_offset
        for i in range(len(groups_melted))
    ]

    # Figure dims
    # (factoring in group widths, space between groups, and initial buffer).
    figwidth = 5
    figheight = (
        level_offset * 2 +
        len(groups_melted) * group_width +
        (len(groups_melted) - 1) * group_spacing
    )
    figheight = min(figheight, figwidth - 1)

    # -- Plot ----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    # colors = sns.color_palette("husl", len(levels_melted))
    from src.colors import get_palette
    colors = get_palette("husl", n_colors=len(levels_melted))

    level_means = []
    for i, group in enumerate(groups_melted):
        group_data = melted_resampling_precis[
            melted_resampling_precis['group'] == group
        ]

        # Base and level x positions for the group
        # group_pos = i * group_spacing
        group_pos = group_centers[i]
        if len(levels_melted) == 1:
            x_positions = [group_pos]
        else:
            x_positions = [
                group_pos + (group_width / 2) * j
                for j in np.linspace(-1, 1, len(levels_melted))
            ]

        for j, level in enumerate(levels_melted):
            level_data = group_data[group_data['level'] == level]
            color = colors[j]

            # Plot horizontal HDI lines
            ax.hlines(
                y=x_positions[j],
                xmin=level_data['hdil'],
                xmax=level_data['hdiu'],
                color=color,
                alpha=0.7,
                linewidth=2,
                label=level if i == 0 else ""
            )

            # Plot the thicker quantile (25%-75%) portion
            ax.hlines(
                y=x_positions[j],
                xmin=level_data['q25'],
                xmax=level_data['q75'],
                color=color,
                linewidth=4,
                alpha=0.9
            )

            # Plot the mean as points
            ax.scatter(
                x=level_data['mean'],
                y=x_positions[j],
                color=color,
                s=80,
                zorder=5
            )
            ax.barh(
                y=x_positions[j],
                width=level_data['mean'],
                color=color,
                edgecolor="white",
                zorder=-1,
                height=level_offset * .65,
                alpha=0.85
            )
            level_means.append((level_data['mean'].values[0], x_positions[j]))

    # Plot lines between means for each level
    for j in range(len(levels_melted)):
        x_means = [
            level_means[k][0]
            for k in range(j, len(level_means), len(levels_melted))
        ]
        y_positions = [
            level_means[k][1]
            for k in range(j, len(level_means), len(levels_melted))
        ]
        ax.plot(
            x_means,
            y_positions,
            color=colors[j],
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )

    ax.set_yticks(group_centers)

    def set_left_aligned_yticklabels(ax, labels, padding_scale=.25):
        ax.set_yticklabels(labels, rotation=0, va="center", ha="left")

        # Get the longest label's width
        renderer = ax.figure.canvas.get_renderer()
        label_widths = [
            label.get_window_extent(renderer).width
            for label in ax.get_yticklabels()
        ]

        # Align labels left without touching the axis
        if label_widths:
            max_width = max(label_widths)

            # Get padding and convert to inches (figure.dpi)
            padding = max_width * padding_scale / ax.figure.dpi
            for label in ax.get_yticklabels():
                label.set_x(-padding)

    # set_left_aligned_yticklabels(ax, groups_melted)
    ax.set_yticklabels(
        groups_melted,
        rotation=0,
        va="center",
        ha="right"
    )
    # Compress plot slightly
    ymin, ymax = ax.get_ylim()
    padding = level_offset
    ax.set_ylim(ymin - padding, ymax + padding)

    ax.set_xlabel("values")
    ax.set_title("Forest plot")

    # Invert y-axis to have groups from top to bottom
    ax.invert_yaxis()

    # Add reference line
    ax.vlines(
        x=1,
        ymin=ymin,
        ymax=ymax + padding,
        color="gray",
        linewidth=.4,
        linestyle="--",
        alpha=0.8,
        # zorder=1
    )

    if len(levels) == 1:
        ax.grid(axis="y")

    ax.legend(
        loc=9,
        ncol=4,
        borderpad=-0,
        borderaxespad=0,
        # handlelength=0,
        # handleheight=0,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.5)

    # sns.despine(ax=ax, left=True, offset=5, trim=True)

    plt.tight_layout()
    plt.show()

    ## Pairplot

    from scipy.stats import pearsonr

    def corrfunc(x, y, ax=None, **kws):
        """Plot the correlation coefficient on a plot.
        """
        r, _ = pearsonr(x, y)
        ax = ax or plt.gca()
        ax.annotate(
            f'ρ = {r:.2f}',
            xy=(0.025, 1),
            xycoords='axes fraction',
            fontsize=10,
            ha='left',
            va='center'
        )

    def label_diag(x, **kws):
        """Plot the group label on a plot.
        """
        ax = plt.gca()
        ax.annotate(
            x.name,
            xy=(0.5, 1),
            xycoords='axes fraction',
            fontsize=10,
            ha='center',
            va='center'
        )
        ax.set_axis_off()

    df = pd.DataFrame({
        'prop_tenure': [0.0, 0.0, 0.06, 0.38, 0.61, 0.01, 0.10, 0.04, 0.22],
        'prop_12m': [0.0, 0.0, 0.06, 0.38, 0.61, 0.01, 0.10, 0.04, 0.22],
        'prop_6m': [0.0, 0.0, 0.10, 0.25, 0.66, 0.02, 0.12, 0.04, 0.22],
        'cat': ['A', 'A', 'B', 'B', 'C', 'C', 'C', 'B', 'C'],
        'level': ['c', 'c', 'c', 'a', 'b', 'c', 'a', 'a', 'b']
    })

    fig = plt.figure(figsize=(6, 6))
    palette = get_palette("husl", n_colors=3)
    # g = sns.pairplot(df)
    g = sns.PairGrid(
        df,
        hue='cat',
        palette=palette,
        # corner=True
    )
    g.map_lower(
        # sns.regplot,
        sns.scatterplot,
        s=20,
        edgecolor="w",
        linewidth=0.5,
        alpha=0.5
    )
    g.map_diag(
        sns.kdeplot,
        fill=True
    )
    g.map_diag(label_diag)
    g.add_legend(
        title="",
        loc='upper center',
        ncol=3,
        borderpad=1,
        borderaxespad=0,
    )
    legend = g._legend
    for handle, text, color in zip(
        legend.legendHandles,
        legend.get_texts(),
        palette[:3]
    ):
        text.set_color(color)
        handle.set_alpha(0)

    # g.map_lower(corrfunc)
    # Add correlation function in the lower triangle (ignoring hue)
    for i, j in zip(*np.tril_indices_from(g.axes, -1)):
        x_data = df[g.x_vars[j]]
        y_data = df[g.y_vars[i]]
        corrfunc(x_data, y_data, ax=g.axes[i, j])

    # Remove all axis labels
    for ax in g.axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')

    # plt.suptitle("Pairgrid")

    # Turn off frames for the upper triangle
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].set_frame_on(False)

    plt.show()
    plt.close()

    ## Faceted plot

    plt.figure(figsize=(6, 6))
    g = sns.FacetGrid(
        df,
        row='level',
        col='cat',
        row_order=['a', 'b', 'c'],
        col_order=['A', 'B', 'C'],
        # hue='cat',
        palette=get_palette("husl", n_colors=3),
        sharex=True,
        sharey=True,
        margin_titles=True,
        despine=True
    )
    g.map(
        # plt.hist, "prop_tenure", alpha=.85, bins=15,
        sns.kdeplot,
        "prop_tenure",
        fill=True
    )
    # g.figure.subplots_adjust(wspace=0, hspace=0)
    g.set_axis_labels(yvar="Density")
    g.refline(x=df["prop_tenure"].median(), linewidth=.4)
    plt.show()

    ## (1a) Melt first

    data_wide = pd.DataFrame({
        'group': np.tile(['A', 'B', 'C'], 2),
        'Hrs_RNDON': np.random.randint(10, 30, 6),
        'Hrs_RNadmin': np.random.randint(15, 35, 6),
        'Hrs_LPNadmin': np.random.randint(5, 25, 6),
    })
    prefix = "Hrs_"
    staff_types = data_wide.filter(regex=f'^{prefix}').columns.to_list()
    id_vars = ['group']

    # Melt
    df_long = data_wide.melt(
        id_vars=id_vars,
        value_vars=staff_types,    # Columns to melt
        var_name='staff_type',     # Name for the melted variable column
        value_name='hours'         # Name for the melted value column
    )
    # Clean up melted names
    df_long['staff_type'] = df_long['staff_type'].str.replace(
        prefix, '', regex=False
    )
    with pd.option_context('display.max_columns', None):
        display("data_wide", "df_long", globs=globals())

    ## (1b) Pivot second (/invert melted data back to wide format)

    # Pivot (requires aggregation of values col over duplicate indices)
    df_wide = df_long.pivot_table(
        index=id_vars,
        values='hours',              # Values to fill in the pivot table
        columns='staff_type',        # Column to pivot
        aggfunc='sum',
        fill_value=0
    )
    # Turn the multi-index into columns
    df_wide = df_wide.reset_index(drop=False)

    # Restore pivoted column naming patterns and order
    df_wide.columns = pd.Index([
        f'{prefix}{col}' if col not in id_vars else col
        for col in df_wide.columns
    ])
    df_wide = df_wide[id_vars + staff_types]

    with pd.option_context('display.max_columns', None):
        display("df_wide", globs=globals())

    ## (2a) Pivot first

    data_long = pd.DataFrame({
        "date": pd.date_range(
            start="2024-01-01", periods=4, freq="D"
        ).strftime('%Y-%m-%d').tolist() * 3,
        "staff_type": np.repeat(["RNDON", "RNadmin", "LPNadmin"], 4),
        "hours": np.random.randint(0, 40, 12)
    })
    prefix = "Hrs_"
    pivot_col = "date"
    id_vars = ["staff_type"]

    # Pivot
    df_wide = data_long.pivot_table(
        index=id_vars,
        columns=pivot_col,
        values="hours",
        aggfunc="sum",
        fill_value=0
    )
    # Remove multi-index for pivoted column
    df_wide = df_wide.reset_index()

    # Add prefix to pivoted columns
    df_wide.columns = pd.Index([
        f"{prefix}{col}" if col not in id_vars else col
        for col in df_wide.columns
    ])

    # Flatten the multi-index to tuples and keep the first level
    df_wide.columns = [
        col[0] if isinstance(col, tuple) else col  # type: ignore
        for col in df_wide.columns  # type: ignore
    ]

    with pd.option_context('display.max_columns', None):
        display("data_long", "df_wide", globs=globals())

    ## (2b) Melt second (/invert pivoted data back to long format)

    df_long = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=df_wide.columns[1:].tolist(),  # All pivoted columns
        var_name=pivot_col,
        value_name="hours"
    )
    # Remove the prefix to restore the original staff type names
    df_long[pivot_col] = df_long[pivot_col].str.replace(
        prefix, '', regex=False
    )
    with pd.option_context('display.max_columns', None):
        display("df_long", globs=globals())

    ## Invert summary for wide format (explode collected data to columns)

    wide_index = pd.date_range(
        start="2024-01-01", periods=5, freq="D"
    ).strftime('%Y-%m-%d')

    df_collected = pd.DataFrame({
        'group': ['A', 'B', 'C'],
        'values': [
            [10, 12, 14, 16, 18],
            [20, 22, 24, 26, 28],
            [30, 32, 34, 36, 38]
        ],
    })

    # Check all lists in 'values' are of the same length
    lengths_values = df_collected['values'].apply(len)
    if lengths_values.nunique() != 1:
        raise ValueError("all lists in 'values' must have the same length")

    # Check that wide index matches list lengths
    if lengths_values.iloc[0] != len(wide_index):
        raise ValueError(
            "'wide_index' length does not match the length of 'values' lists"
        )

    # Expand the 'values' list into columns, adding date-based column names
    df_expanded = pd.DataFrame(
        df_collected['values'].tolist(), columns=wide_index
    )
    # Concatenate with original df (without the 'values' column)
    inverse_df_wide = pd.concat(
        [df_collected[['group']].reset_index(drop=True), df_expanded], axis=1
    )
    with pd.option_context('display.max_columns', None):
        display(
            "df_collected", "inverse_df_wide", globs=globals()
        )
