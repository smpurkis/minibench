import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import math


def make_radar_chart(
    data_list: list[dict[str, float]],
    labels: list[str],
    save_path: Optional[Path] = None,
    method: str = "matplotlib",
) -> Optional[go.Figure]:
    if method == "matplotlib":
        # Number of variables
        num_vars = len(data_list[0])

        # Compute angle each axis will occupy
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # The plot is a circle, so we need to "complete the loop"
        # and append the start value to the end.
        angles += angles[:1]

        # Create figure and polar subplot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Set y-axis limits to always be between 0 and 1
        ax.set_ylim([0, 1])

        # Assign a unique color and line style to each chart
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
        linestyles = ["-", "--", "-.", ":"]
        random.shuffle(linestyles)

        # Loop through each data dictionary and plot on the same subplot
        keys = sorted(list(data_list[0].keys()))
        for i, data in enumerate(data_list):
            values = [data[k] for k in keys]
            values += values[:1]
            ax.plot(
                angles,
                values,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                label=labels[i],
            )

        # Draw labels for the axes
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([k.split("/")[0] for k in keys])

        # Add a legend below the plot
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=2)

        # Adjust figure size and margins to include the legend
        fig.subplots_adjust(bottom=0.26)

        # Save the plot as a PNG file
        if save_path is not None:
            plt.savefig(save_path.with_suffix(".png"))
    elif method == "plotly":
        # Extract keys and values from data dictionaries
        keys = sorted(list(data_list[0].keys()))
        values_list = [[data[k] for k in keys] for data in data_list]

        # Create figure
        fig = go.Figure()

        # sort labels with values_list
        sv_list = sorted(zip(labels, values_list), key=lambda x: x[1][0])
        labels = [x[0] for x in sv_list]
        values_list = [x[1] for x in sv_list]

        # Add traces for each data dictionary
        for i, values in enumerate(values_list):
            fig.add_trace(
                go.Scatterpolar(
                    r=values
                    + values[:1],  # Add the first value to the end to close the shape
                    theta=[k.split("/")[0] for k in keys]
                    + [
                        keys[0].split("/")[0]
                    ],  # Repeat the first label to close the shape
                    fill=None,
                    name=labels[i],
                    line=dict(),
                )
            )

        # Set layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    range=[0, 1.1],
                    visible=True,
                    color="black",
                    showline=True,
                    linewidth=2,
                    gridcolor="black",
                    gridwidth=1,
                ),
                angularaxis=dict(
                    showticklabels=True,
                    tickangle=0,
                    color="black",
                    showline=True,
                    linewidth=2,
                ),
            ),
            showlegend=True,
            legend=dict(x=0, y=1.5, font=dict(size=8)),
            margin=dict(l=50, r=50, t=50, b=50),
            width=800,
            height=800,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        # Add an axis line for each category
        # for key in keys:
        #     fig.add_shape(
        #         type="line",
        #         xref="paper",
        #         yref="paper",
        #         x0=0.5,
        #         y0=0.5,
        #         x1=0.5
        #         + 0.4 * math.cos(math.radians(keys.index(key) * 360 / len(keys))),
        #         y1=0.5
        #         + 0.4 * math.sin(math.radians(keys.index(key) * 360 / len(keys))),
        #         line=dict(color="Black", width=1),
        #     )

        # Save the plot as a PNG file
        if save_path is not None:
            fig.write_image(str(save_path.with_suffix(".png")))
        return fig


if __name__ == "__main__":
    make_radar_chart(
        data_list=[
            {
                "A": 0.2,
                "B": 0.4,
                "C": 0.6,
                "D": 0.8,
                "E": 1.0,
            },
            {
                "E": 0.8,
                "B": 0.3,
                "C": 0.6,
                "A": 0.4,
                "D": 0.7,
            },
        ],
        labels=["First", "Second"],
        save_path=Path("radar_chart"),
    )
