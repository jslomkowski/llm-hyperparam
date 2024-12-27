import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Set the plotting style
plt.style.use("seaborn-v0_8-white")

# Standard plot parameters
FIGURE_SIZE = (10, 6)
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
GRID_STYLE = {"axis": "y", "linestyle": "--", "alpha": 0.7}
BAR_ALPHA = 0.7
BAR_EDGE_COLOR = "black"

# Load data
adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = adult.target


# Plotting functions
def plot_numerical_column(column, top_n=None, bin_size=None, start=None, end=None):
    min_val = column.min()
    max_val = column.max()
    start = start if start is not None else min_val
    end = end if end is not None else max_val

    if bin_size is not None:
        bins = np.arange(start, end + bin_size, bin_size)
    else:
        bins = 30

    counts, bin_edges = np.histogram(column.dropna(), bins=bins)
    bin_labels = [
        f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_edges) - 1)
    ]
    bin_data = list(zip(bin_labels, counts))

    if top_n is not None:
        bin_data_sorted = sorted(bin_data, key=lambda x: x[1], reverse=True)[:top_n]
    else:
        bin_data_sorted = bin_data

    top_bins = [x[0] for x in bin_data_sorted]
    top_counts = [x[1] for x in bin_data_sorted]

    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(top_bins, top_counts, alpha=BAR_ALPHA, edgecolor=BAR_EDGE_COLOR)
    plt.title(
        f"Top {top_n if top_n else 'all'} bins of {column.name}", fontsize=TITLE_SIZE
    )
    plt.xlabel("Bins", fontsize=LABEL_SIZE)
    plt.ylabel("Count", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_categorical_column(column, top_n=None):
    value_counts = column.value_counts(dropna=False)
    if top_n is not None:
        value_counts = value_counts.head(top_n)
    value_counts.index = value_counts.index.astype(str)
    plt.figure(figsize=FIGURE_SIZE)
    plt.bar(
        value_counts.index,
        value_counts.values,
        alpha=BAR_ALPHA,
        edgecolor=BAR_EDGE_COLOR,
    )
    plt.title(f"{column.name} Column Bar Plot", fontsize=TITLE_SIZE)
    plt.xlabel(f"{column.name} Categories", fontsize=LABEL_SIZE)
    plt.ylabel("Count", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_two_numerical_columns(column_x, column_y, trend_line=False):
    plt.figure(figsize=FIGURE_SIZE)
    plt.scatter(
        column_x,
        column_y,
        alpha=BAR_ALPHA,
        edgecolors=BAR_EDGE_COLOR,
        label="Data Points",
    )
    if trend_line:
        coefficients = np.polyfit(column_x, column_y, 1)
        poly_eq = np.poly1d(coefficients)
        trend_y = poly_eq(column_x)
        plt.plot(
            column_x,
            trend_y,
            linewidth=2,
            label="Trend Line",
        )
    plt.title(f"{column_x.name} vs. {column_y.name}", fontsize=TITLE_SIZE)
    plt.xlabel(f"{column_x.name}", fontsize=LABEL_SIZE)
    plt.ylabel(f"{column_y.name}", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.ticklabel_format(style="plain", axis="x")
    plt.ticklabel_format(style="plain", axis="y")
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_categorical_vs_numerical(column_cat, column_num, top_n=None):
    column_num_cleaned = column_num[pd.to_numeric(column_num, errors="coerce").notna()]
    column_cat_cleaned = column_cat[column_num_cleaned.index]
    if top_n is not None:
        top_categories = column_cat_cleaned.value_counts().head(top_n).index
        column_cat_cleaned = column_cat_cleaned[column_cat_cleaned.isin(top_categories)]
        column_num_cleaned = column_num_cleaned[column_cat_cleaned.index]

    data = [
        column_num_cleaned[column_cat_cleaned == category]
        for category in column_cat_cleaned.unique()
    ]
    plt.figure(figsize=FIGURE_SIZE)
    bp = plt.boxplot(
        data,
        labels=column_cat_cleaned.unique(),
        patch_artist=True,
    )
    # Set alpha for boxplots
    for patch in bp["boxes"]:
        patch.set_alpha(BAR_ALPHA)
    plt.title(f"{column_num.name} by {column_cat.name}", fontsize=TITLE_SIZE)
    plt.xlabel(f"{column_cat.name}", fontsize=LABEL_SIZE)
    plt.ylabel(f"{column_num.name}", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_two_categorical_columns(
    column_x, column_y, top_n=None, show_values=True, as_percent=False
):
    cross_tab = pd.crosstab(column_x, column_y)
    if top_n is not None:
        top_categories_x = column_x.value_counts().head(top_n).index
        top_categories_y = column_y.value_counts().head(top_n).index
        cross_tab = cross_tab.loc[
            cross_tab.index.intersection(top_categories_x),
            cross_tab.columns.intersection(top_categories_y),
        ]
    if as_percent:
        cross_tab = (cross_tab / cross_tab.values.sum()) * 100
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(cross_tab, cmap="Blues", aspect="auto")
    plt.xticks(
        range(len(cross_tab.columns)),
        cross_tab.columns,
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    plt.yticks(
        range(len(cross_tab.index)),
        cross_tab.index,
        fontsize=TICK_LABEL_SIZE,
    )
    plt.title(f"{column_x.name} vs. {column_y.name}", fontsize=TITLE_SIZE)
    plt.xlabel(f"{column_y.name}", fontsize=LABEL_SIZE)
    plt.ylabel(f"{column_x.name}", fontsize=LABEL_SIZE)
    if show_values:
        for i in range(len(cross_tab.index)):
            for j in range(len(cross_tab.columns)):
                value = (
                    f"{cross_tab.iloc[i, j]:.1f}%"
                    if as_percent
                    else f"{cross_tab.iloc[i, j]}"
                )
                plt.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=TICK_LABEL_SIZE,
                )
    plt.tight_layout()
    plt.show()


def plot_time_series(*series_list):
    if len(series_list) == 0:
        raise ValueError("At least one time series must be provided.")
    plt.figure(figsize=FIGURE_SIZE)
    for i, series in enumerate(series_list):
        plt.plot(
            series.index,
            series.values,
            linestyle="-" if i == 0 else "--",
            alpha=BAR_ALPHA,
            label=series.name if series.name else f"Series {i + 1}",
        )
    if len(series_list) > 1:
        title = " and ".join(
            [
                series.name if series.name else f"Series {i + 1}"
                for i, series in enumerate(series_list)
            ]
        )
        plt.title(title, fontsize=TITLE_SIZE)
    else:
        plt.title(
            series_list[0].name if series_list[0].name else "Time Series",
            fontsize=TITLE_SIZE,
        )
    plt.xlabel("Time", fontsize=LABEL_SIZE)
    plt.ylabel("Value", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.legend(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def generate_random_time_series(start_date, end_date, freq="D", seed=None):
    if seed is not None:
        np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    values = np.random.randn(len(date_range)).cumsum()
    return pd.Series(data=values, index=date_range, name="Random Time Series")


# Example usage
plot_numerical_column(X["age"], bin_size=5, start=0, end=100)
plot_categorical_column(X["workclass"])
plot_two_numerical_columns(X["age"], X["education-num"], trend_line=True)
plot_two_categorical_columns(X["workclass"], X["marital-status"])
plot_categorical_vs_numerical(X["workclass"], X["age"])

ts1 = generate_random_time_series(
    start_date="2023-01-01", end_date="2023-11-30", freq="D", seed=42
)
ts2 = generate_random_time_series(
    start_date="2023-12-01", end_date="2023-12-31", freq="D", seed=42
)
ts3 = generate_random_time_series(
    start_date="2023-12-01", end_date="2023-12-31", freq="D", seed=43
)

plot_time_series(ts1, ts2, ts3)
