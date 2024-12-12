# %%
# Configure

# Python interactive window help:
# https://code.visualstudio.com/docs/python/jupyter-support-py

# flake8: noqa: E403

import re
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from pathlib import Path
from pandas.plotting import scatter_matrix
from great_tables import GT, md, nanoplot_options

src_path = Path('..')
sys.path.append(str(src_path.resolve()))

from src.paths import get_path_to
from src.precis import make_precis
from src.inspection import display
from src.stylesheet import customize_plots
from src.inspection import make_df, display, display2

customize_plots()
# %config InlineBackend.figure_format = 'svg'

# %%
# Load data
if 'data' not in locals():
    data = pd.read_csv(
        get_path_to("data", "raw", "PBJ_Daily_Nurse_Staffing_Q1_2024.zip"),
        encoding='ISO-8859-1',
        low_memory=False
    )
else:
    print("data loaded.")

# %%
# Visualize relationships

attributes = ["Hrs_RN", "Hrs_LPN_ctr", "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"]
n = len(attributes)

fig, axs = plt.subplots(n, n, figsize=(6, 6))
scatter_matrix(
    data[attributes].sample(200),
    ax=axs, alpha=.7,
    hist_kwds=dict(bins=15, linewidth=0)
)
fig.align_ylabels(axs[:, 0])
fig.align_xlabels(axs[-1, :])
for ax in axs.flatten():
    ax.tick_params(axis='both', which='both', length=3.5)

# save_fig("scatter_matrix_plot")
plt.show()

# %%
# Tests

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
id_cols = ["group", "level"]
stub_patterns   = r"(_coefficient_\d+|_parameter_\d+)"
suffix_patterns = r"(coefficient_\d+|parameter_\d+)"

stubnames = [
    col for col in resampling_precis.columns
    .str.replace(stub_patterns, '', regex=True).unique()
    if col not in id_cols
]

# Verify no missing columns between stubs and df
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

## Prettify

precis_nanofmt = resampling_precis.copy()
precis_nanofmt[data_cols[0]] = precis_nanofmt[data_cols[0]].apply(
    lambda x: {"val": x}
)
pretty_resampling_precis = (
    GT(precis_nanofmt.reset_index())
    .tab_header(
        title="Accuracy scores",
        subtitle="With 100 rounds of cross-validation for the top 10 models",
    )
    .tab_stub(rowname_col="level", groupname_col="group")
    .tab_spanner(
        label="mean",
        columns=mean_cols
    )
    .tab_spanner(
        label="std",
        columns=stddev_cols
    )
    .tab_spanner(
        label="size",
        columns=size_cols
    )
    .tab_spanner(
        label="error",
        columns=err_cols
    )
    .fmt_integer(columns=size_cols)
    .fmt_number(columns=mean_cols + stddev_cols + err_cols, decimals=2)
    .cols_hide(columns=data_cols[1:])
    .fmt_nanoplot(data_cols[0], plot_type="bar")  # or "line"
    .cols_move_to_end(data_cols[0])
    # .tab_source_note(source_note="Source: .")
    # .tab_source_note(
    #     source_note=md("Reference: Bain, M. T. (2024) *Title*. Source.")
    # )
# ).save(
#     file=""
)
pretty_resampling_precis

# %%
