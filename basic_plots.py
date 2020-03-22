from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from bokeh.layouts import gridplot
from bokeh.models import HoverTool, Title
from bokeh.palettes import brewer
from bokeh.plotting import figure, ColumnDataSource
from bokeh.transform import dodge


colors = brewer["Set1"]
shades = brewer['RdYlBu']


def line_plot(
    index: np.array,
    serie: np.array,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = colors[3][1],
) -> figure:
    """Generate a line plot of a serie.

    :param index: x-axis range.
    :param serie: y_axis range.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param color: color of the line.
    :param source: data source.
    :return: a line figure.
    """
    # Create figure
    plot = figure(title=title,
                  tools=["save", 'crosshair', 'wheel_zoom', 'pan', 'reset'],
                  background_fill_color="#fafafa")

    # Create line
    plot.line(index, serie, color=color)

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.grid.grid_line_color = "#fafafa"

    return plot


def scatter_plot(
    serie_1: np.array,
    serie_2: np.array,
    size: int,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = colors[3][1],
    alpha: float = 1,
    hover: bool = False,
    tags: np.array = None,
) -> figure:
    """Generate a scatter plot of a serie.

    :param serie_1: x-axis range.
    :param serie: y_axis range.
    :param size: size of circles.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param color: color of the cicle.
    :param hover: activate the HoverTool.
    :return: a scatter plot figure.
    """
    # Create source (bokeh spe)
    if tags is None:
        tags = serie_1

    source = ColumnDataSource({"serie_1": serie_1, "serie_2": serie_2, "tags": tags})

    # Create figure
    plot = figure(title=title,
                  tools=["save", 'crosshair', 'wheel_zoom', 'pan', 'reset'],
                  background_fill_color="#fafafa")

    # Create circles
    plot.circle(x="serie_1", y="serie_2", size=size, color=color, fill_alpha=alpha, source=source)

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.grid.grid_line_color = "#fafafa"

    if hover:
        # Hover tool
        hover = HoverTool(tooltips=[('Tags ', '@tags'), ('Value', '@value')])
        plot.add_tools(hover)

    return plot


def frequency_plot(
    serie: np.array,
    n_bins: int,
    bins_range: Optional[Tuple[float, float]] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    edge_ticker: bool = False,
    color: str = colors[3][1],
    step_mode: bool = False,
    density: bool = False,
) -> figure:
    """Generate a bar plot(histogram) of a serie.

    :param serie: y_axis range.
    :param n_bins: number of equal-width bins in the given range.
    :param bins_range: the lower and upper range of the bins.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param edge_ticker: use histogram eges as x-axis range.
    :param color: color of the line.
    :param step_mode: use lines (step) and not bar.
    :param density: normalized such that the integral over the range is 1.
    :return: a histogram figure.
    """
    # Compute histogram
    hist, edges = np.histogram(serie, n_bins, density=density, range=bins_range)
    step = abs((min(edges)-max(edges))/n_bins)

    # Create figure
    plot = figure(title=title, tools=["save", 'crosshair'],
                  background_fill_color="#fafafa")

    # Use step mode
    if step_mode:
        plot.step(x=edges[:-1], y=hist, color=color)

    # Use bar mode
    else:
        left = edges[:-1] + step*0.1
        right = edges[1:] - step*0.1
        plot.quad(left=left, right=right, top=hist, bottom=0, color=color)

    # Plot parameters
    plot.xaxis.axis_label = x_label
    if edge_ticker:
        plot.xaxis.ticker = edges
    plot.yaxis.axis_label = y_label
    plot.background_fill_color = "#fefefe"

    return plot


def multiline_plot(
    indexes: List[np.array],
    series: List[np.array],
    names: List[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: List[str] = colors[9],
    legend_location: str = "top_right"
) -> figure:
    """Generate line plots of several series.

    :param indexes: x-axis ranges.
    :param series: y_axis ranges.
    :param names: names for legend of each serie.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param colors: colors of the different lines.
    :param legend_location: location of the legend.
    :return: a line figure.
    """
    # Create figure
    plot = figure(title=title,
                  tools=["save", 'crosshair', 'wheel_zoom', 'pan', 'reset'],
                  background_fill_color="#fafafa")

    # Create lines
    for k in range(len(series)):
        plot.line(indexes[k], series[k], color=colors[k], legend_label=names[k])

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.background_fill_color = "#fefefe"
    plot.legend.background_fill_color = "#fafafa"
    plot.legend.location = legend_location
    plot.legend.click_policy = "hide"
    plot.grid.grid_line_color = "#fafafa"

    return plot


def multifrequency_plot(
    series: List[np.array],
    names: List[str],
    n_bins: int,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    edge_ticker: bool = False,
    color: List[str] = colors,
    step_mode: bool = True,
    density: bool = False,
    legend_location: str = "top_right"
) -> figure:
    """Generate a bar plot(histogram) of a serie.

    :param series: y_axis ranges.
    :param names: names for legend of each serie.
    :param n_bins: number of equal-width bins in the given range.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param edge_ticker: use histogram eges as x-axis range.
    :param colors: colors of glyphs.
    :param step_mode: use lines (step) and not bar.
    :param density: normalized such that the integral over the range is 1.
    :param legend_location: location of the legend.
    :return: a histogram figure.
    """
    # Create figure
    plot = figure(title=title, tools=["save", 'crosshair'],
                  background_fill_color="#fafafa")

    for k, serie in enumerate(series):
        # Compute histogram
        hist, edges = np.histogram(serie, n_bins, density=density)
        step = abs((min(edges)-max(edges))/n_bins)

        # Use step mode
        if step_mode:
            plot.step(x=edges[:-1], y=hist, color=colors[9][k], legend_label=names[k])

        # Use bar mode
        else:
            left = edges[:-1] + step*0.1
            right = edges[1:] - step*0.1
            plot.quad(left=left, right=right, top=hist, bottom=0,
                      color=color[k], legend_label=names[k])

    # Plot parameters
    plot.xaxis.axis_label = x_label
    if edge_ticker:
        plot.xaxis.ticker = edges
    plot.yaxis.axis_label = y_label
    plot.background_fill_color = "#fefefe"
    plot.legend.background_fill_color = "#fafafa"
    plot.legend.location = legend_location
    plot.legend.click_policy = "hide"

    return plot


def multiscatter_plot(
    indexes: List[np.array],
    series: List[np.array],
    names: List[str],
    size: int,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: List[str] = colors,
    alpha: float = 1,
    legend_location: str = "top_right"
) -> figure:
    """Generate a scatter plot of several series.

    :param indexes: x-axis ranges.
    :param series: y_axis ranges.
    :param names: names for legend of each serie.
    :param size: size of circles.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param colors: color of the circle.
    :param alpha: alpha of the cirlce.
    :param legend_location: location of the legend.
    :return: a scatter plot figure.
    """
    # Create figure
    plot = figure(title=title,
                  tools=["save", 'crosshair', 'wheel_zoom', 'pan', 'reset'],
                  background_fill_color="#fafafa")

    for k, serie in enumerate(series):
        # Create circles
        plot.circle(x=indexes[k], y=serie, size=size, color=colors[9]
                    [k], legend_label=names[k], fill_alpha=alpha)

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.background_fill_color = "#fefefe"
    plot.legend.background_fill_color = "#fafafa"
    plot.legend.location = legend_location
    plot.legend.click_policy = "hide"

    return plot


def categorical_plot(
    categories: List[Any],
    values: List[Any],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color: str = colors[3][1],
    alpha: float = 1,
    vertical: bool = False
) -> figure:
    """Generate a bar plot of categorical data.

    :param categories: list of categories of data.
    :param values: list of values for each category.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param color: color of the line.
    :param vertical: generate a vertical plot.
    :return: a categorical bar chart.
    """
    # Create source (bokeh spe)
    source = ColumnDataSource({"categories": categories, "value": values})

    # Plot a vertical chart
    if vertical:
        # Create figure
        plot = figure(y_range=categories, title=title,
                      background_fill_color="#fafafa")
        # Create bars
        plot.hbar(y="categories", right="value", height=0.9,
                  color=color, fill_alpha=alpha, hatch_alpha=alpha, source=source)

    # Plot a horizontal chart
    else:
        # Create figure
        plot = figure(x_range=categories, title=title, background_fill_color="#fafafa")
        # Create bars
        plot.vbar(x="categories", top="value", width=0.9,
                  color=color, fill_alpha=alpha, hatch_alpha=alpha, source=source)
        # Ticker orientation
        plot.xaxis.major_label_orientation = np.pi/4

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.grid.grid_line_color = "#fafafa"

    # Hover tool
    hover = HoverTool(tooltips=[('Category ', '@categories'), ('Value', '@value')])
    plot.add_tools(hover)

    return plot


def multicategorical_plot(
    categories: List[Any],
    values: Dict[str, List[Any]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: str = colors[9],
    alpha: float = 1,
    vertical: bool = False
) -> figure:
    """Generate a bar plot of categorical data.

    :param categories: list of categories of data.
    :param values: list of list of values for each subcategory.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param colors: color of the line.
    :param vertical: generate a vertical plot.
    :return: a categorical bar chart.
    """
    # Create source (bokeh spe)
    source = ColumnDataSource({**{"categories": categories}, **values})

    n_subcategories = len(values) - 1
    width = (0.25 * n_subcategories) / 2
    strides = np.arange(-width, width+0.25, 0.25)

    # Plot a vertical chart
    if vertical:
        # Create figure
        plot = figure(y_range=categories, title=title, background_fill_color="#fafafa")
        # Create bars
        for k, name in enumerate(values):
            plot.hbar(y=dodge("categories", strides[k], range=plot.y_range), right=name, height=0.2,
                      color=colors[k], fill_alpha=alpha, hatch_alpha=alpha, source=source,
                      legend_label=name)

    # Plot a horizontal chart
    else:
        # Create figure
        plot = figure(x_range=categories, title=title, background_fill_color="#fafafa")
        # Create bars
        for k, name in enumerate(values):
            plot.vbar(x=dodge("categories", strides[k], range=plot.x_range), top=name, width=0.2,
                      color=colors[k], fill_alpha=alpha, hatch_alpha=alpha, source=source,
                      legend_label=name)
        # Ticker orientation
        plot.xaxis.major_label_orientation = np.pi/4

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.background_fill_color = "#fefefe"
    plot.legend.background_fill_color = "#fafafa"
    plot.legend.click_policy = "hide"
    plot.grid.grid_line_color = "#fafafa"

    # Hover tool
    hover = HoverTool(tooltips=[('Category ', '@categories')])
    plot.add_tools(hover)

    return plot


def stacked_area_plot(
    index: List[float],
    series: List[List[float]],
    names: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    y_max: float = 1.0,
    colors: List[str] = shades
) -> figure:
    """Generate a stacked area plot.

    :param index: x-axis range.
    :param series: y_axis ranges.
    :param names: names for legend of each serie.
    :param title: title of the chart.
    :param x_label: x_axis label.
    :param y_label: y_axis label.
    :param colors: color of the areas (should have at least 3 series by default).
    :return: a stacked area plot figure.
    """
    # Create source (bokeh spe)
    source = ColumnDataSource(pd.DataFrame(np.transpose(
        np.array(series))).set_index(np.array(index)).add_prefix('y'))

    # Create figure
    plot = figure(title=title, tools=["save"], x_range=(
        np.min(index), np.max(index)), y_range=(0, y_max))

    # Create areas
    y_names = ["y%d" % i for i in range(len(names))]
    plot.varea_stack(stackers=y_names, x='index',
                     color=colors[len(names)], source=source, legend_label=names)

    # Plot parameters
    plot.xaxis.axis_label = x_label
    plot.yaxis.axis_label = y_label
    plot.grid.grid_line_color = "#fafafa"
    plot.legend.items.reverse()

    return plot


def scatter_matrix_plot(
    series: List[List[int]],
    names: List[str],
    n_bins: List[int],
    size: int,
    density: bool = False,
    plot_width=300,
    plot_height=300,
) -> figure:
    """Generate a scatter-matrix from given series.

    :param series: data points.
    :param names: names for legend of each serie.
    :param n_bins: number of histogram bins for each series.
    :param size: size of circles for scatter plots.
    :param density: normalized histograms such that the integral over the range is 1.
    :param plot_width: width of the figure.
    :param plot_height: height of the figure.
    :return: scatter-matrix plot.
    """
    n_series = len(series)
    grid_list = np.array([[None] * n_series] * n_series)

    for idx_1 in range(n_series):
        serie_1 = series[idx_1]
        # Generate histograms (diagonal charts)
        grid_list[idx_1][idx_1] = frequency_plot(serie_1, n_bins=n_bins[idx_1], density=density)

        for idx_2 in range(idx_1+1, n_series):
            serie_2 = series[idx_2]
            # Generate scatters plots related to `serie_1` and `serie_2`
            grid_list[idx_2][idx_1] = scatter_plot(serie_1, serie_2, size=size)
            grid_list[idx_1][idx_2] = scatter_plot(serie_2, serie_1, size=size)

        # Display names of series
        grid_list[idx_1][0].add_layout(Title(text=names[idx_1], align="center"), "left")
        grid_list[-1][idx_1].add_layout(Title(text=names[idx_1], align="center"), "below")

    return gridplot(grid_list.tolist(), plot_width=plot_width, plot_height=plot_height)
