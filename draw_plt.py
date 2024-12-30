import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

plt.style.use("seaborn-v0_8-white")

FIGURE_SIZE = (10, 6)
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
BAR_ALPHA = 0.7
BAR_EDGE_COLOR = "black"
GRID_STYLE = {"axis": "y", "linestyle": "--", "alpha": 0.7}
BAR_WIDTH_FACTOR = 0.8

COLOR_PALETTE = plt.get_cmap("RdYlGn")


def get_colors(n):
    return COLOR_PALETTE(np.linspace(0, 1, n))


adult = fetch_openml("adult", version=2, as_frame=True)
X = adult.data
y = adult.target


def plot_numerical_column(
    column,
    by=None,
    top_n=None,
    bin_size=None,
    start=None,
    end=None,
    as_percentage=False,
):
    min_val = column.min()
    max_val = column.max()
    start = start if start is not None else min_val
    end = end if end is not None else max_val

    if bin_size is not None:
        bins = np.arange(start, end + bin_size, bin_size)
    else:
        bins = 30

    plt.figure(figsize=FIGURE_SIZE)

    if by is not None:
        grouped = column.groupby(by, observed=False)
        n_groups = len(grouped)
        bar_width = BAR_WIDTH_FACTOR / n_groups
        colors = get_colors(n_groups)

        for i, (name, group) in enumerate(grouped):
            counts, bin_edges = np.histogram(group.dropna(), bins=bins)
            bin_labels = [
                f"[{bin_edges[j]:.2f}, {bin_edges[j+1]:.2f})"
                for j in range(len(bin_edges) - 1)
            ]
            x_positions = np.arange(len(bin_labels)) + i * bar_width
            if as_percentage:
                counts = (counts / counts.sum()) * 100
            plt.bar(
                x_positions,
                counts,
                width=bar_width,
                alpha=BAR_ALPHA,
                color=colors[i],
                edgecolor=BAR_EDGE_COLOR,
                label=str(name),
            )

        plt.xticks(
            np.arange(len(bin_labels)) + bar_width * (n_groups - 1) / 2,
            bin_labels,
            rotation=45,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        plt.legend(title=by.name, fontsize=TICK_LABEL_SIZE)
        plt.title(f"Distribution of {column.name} by {by.name}", fontsize=TITLE_SIZE)

    else:
        counts, bin_edges = np.histogram(column.dropna(), bins=bins)
        bin_labels = [
            f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
            for i in range(len(bin_edges) - 1)
        ]
        if as_percentage:
            counts = (counts / counts.sum()) * 100
        plt.bar(
            bin_labels,
            counts,
            alpha=BAR_ALPHA,
            color=get_colors(1)[0],
            edgecolor=BAR_EDGE_COLOR,
        )
        plt.title(f"Distribution of {column.name}", fontsize=TITLE_SIZE)

    plt.xlabel("Bins", fontsize=LABEL_SIZE)
    ylabel = "Percentage" if as_percentage else "Count"
    plt.ylabel(ylabel, fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_categorical_column(column, by=None, top_n=None):
    plt.figure(figsize=FIGURE_SIZE)

    if by is not None:
        grouped = column.groupby(by, observed=False)
        n_groups = len(grouped)
        bar_width = BAR_WIDTH_FACTOR / n_groups
        colors = get_colors(n_groups)

        for i, (name, group) in enumerate(grouped):
            value_counts = group.value_counts(dropna=False)
            if top_n is not None:
                value_counts = value_counts.head(top_n)
            value_counts.index = value_counts.index.astype(str)
            x_positions = np.arange(len(value_counts.index)) + i * bar_width

            plt.bar(
                x_positions,
                value_counts.values,
                width=bar_width,
                alpha=BAR_ALPHA,
                color=colors[i],
                edgecolor=BAR_EDGE_COLOR,
                label=str(name),
            )

        plt.xticks(
            np.arange(len(value_counts.index)) + bar_width * (n_groups - 1) / 2,
            value_counts.index,
            rotation=45,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        plt.legend(title=by.name, fontsize=TICK_LABEL_SIZE)
        plt.title(f"{column.name} Column Bar Plot by {by.name}", fontsize=TITLE_SIZE)

    else:
        value_counts = column.value_counts(dropna=False)
        if top_n is not None:
            value_counts = value_counts.head(top_n)
        value_counts.index = value_counts.index.astype(str)

        plt.bar(
            value_counts.index,
            value_counts.values,
            alpha=BAR_ALPHA,
            color=get_colors(1)[0],
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


def plot_two_numerical_columns(column_x, column_y, by=None, trend_line=False):
    plt.figure(figsize=FIGURE_SIZE)

    if by is not None:
        grouped = column_x.groupby(by, observed=False)
        n_groups = len(grouped)
        colors = get_colors(n_groups)

        for i, (name, group) in enumerate(grouped):
            plt.scatter(
                group,
                column_y.loc[group.index],
                alpha=BAR_ALPHA,
                edgecolors=BAR_EDGE_COLOR,
                color=colors[i],
                label=str(name),
            )
        plt.legend(title=by.name, fontsize=TICK_LABEL_SIZE)

    else:
        plt.scatter(
            column_x,
            column_y,
            alpha=BAR_ALPHA,
            edgecolors=BAR_EDGE_COLOR,
            color=get_colors(1)[0],
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
            color="red",
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


def plot_categorical_vs_numerical(column_cat, column_num, by=None, top_n=None):
    plt.figure(figsize=FIGURE_SIZE)

    column_num_cleaned = column_num[pd.to_numeric(column_num, errors="coerce").notna()]
    column_cat_cleaned = column_cat[column_num_cleaned.index]

    if top_n is not None:
        top_categories = column_cat_cleaned.value_counts().head(top_n).index
        column_cat_cleaned = column_cat_cleaned[column_cat_cleaned.isin(top_categories)]
        column_num_cleaned = column_num_cleaned[column_cat_cleaned.index]

    if by is not None:
        by_cleaned = by[column_cat_cleaned.index].dropna()
        unique_by = by_cleaned.unique()
        n_groups = len(unique_by)

        unique_cat = column_cat_cleaned.unique()
        x_positions = np.arange(len(unique_cat))
        bar_width = BAR_WIDTH_FACTOR / n_groups
        colors = get_colors(n_groups)

        legend_patches = []

        for i, sub_cat in enumerate(unique_by):
            mask = by_cleaned == sub_cat
            cat_data = column_cat_cleaned[mask]
            num_data = column_num_cleaned[mask]

            data_for_boxplot = []
            for single_cat in unique_cat:
                data_for_boxplot.append(num_data[cat_data == single_cat])

            bp = plt.boxplot(
                data_for_boxplot,
                positions=x_positions + i * bar_width,
                widths=bar_width,
                patch_artist=True,
                labels=[None] * len(unique_cat),
            )

            for box_patch in bp["boxes"]:
                box_patch.set_facecolor(colors[i])
                box_patch.set_alpha(BAR_ALPHA)
                box_patch.set_edgecolor(BAR_EDGE_COLOR)

            legend_patches.append(
                mpatches.Patch(color=colors[i], alpha=BAR_ALPHA, label=sub_cat)
            )

        plt.xticks(
            x_positions + bar_width * (n_groups - 1) / 2,
            unique_cat,
            rotation=45,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        plt.legend(handles=legend_patches, title=by.name, fontsize=TICK_LABEL_SIZE)
        plt.title(
            f"{column_num.name} by {column_cat.name} and {by.name}", fontsize=TITLE_SIZE
        )

    else:
        unique_cat = column_cat_cleaned.unique()
        data_for_boxplot = [
            column_num_cleaned[column_cat_cleaned == cat] for cat in unique_cat
        ]
        bp = plt.boxplot(
            data_for_boxplot,
            patch_artist=True,
            labels=unique_cat,
        )

        for box_patch in bp["boxes"]:
            box_patch.set_alpha(BAR_ALPHA)
            box_patch.set_edgecolor(BAR_EDGE_COLOR)
            box_patch.set_facecolor(get_colors(1)[0])

        plt.title(f"{column_num.name} by {column_cat.name}", fontsize=TITLE_SIZE)

    plt.xlabel(f"{column_cat.name} Categories", fontsize=LABEL_SIZE)
    plt.ylabel(f"{column_num.name}", fontsize=LABEL_SIZE)
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    plt.grid(**GRID_STYLE)
    plt.tight_layout()
    plt.show()


def plot_time_series(*series_list):
    plt.figure(figsize=FIGURE_SIZE)
    colors = get_colors(len(series_list))

    for i, series in enumerate(series_list):
        plt.plot(
            series.index,
            series.values,
            alpha=BAR_ALPHA,
            color=colors[i],
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
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    values = np.random.randn(len(date_range)).cumsum()
    return pd.Series(data=values, index=date_range, name="Random Time Series")


plot_numerical_column(X["age"], by=X["sex"], bin_size=5, start=0, end=100)
plot_categorical_column(X["workclass"], by=X["sex"], top_n=5)
plot_two_numerical_columns(X["age"], X["education-num"], by=X["sex"], trend_line=True)
plot_two_categorical_columns(X["workclass"], X["marital-status"])
plot_categorical_vs_numerical(X["workclass"], X["age"], by=X["sex"], top_n=None)

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
