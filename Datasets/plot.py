import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# def make_radar_chart(data: dict[str, float], save_path: Path) -> None:
#     # Number of variables
#     num_vars = len(data)

#     # Compute angle each axis will occupy
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

#     # The plot is a circle, so we need to "complete the loop"
#     # and append the start value to the end.
#     values = list(data.values())
#     values += values[:1]
#     angles += angles[:1]

#     # Create figure and polar subplot
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

#     # Draw the outline of our data.
#     ax.plot(angles, values, color="purple")

#     # Draw labels for the axes
#     ax.set_yticklabels([])
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(list(data.keys()))

#     # Save the plot as a PNG file
#     plt.savefig(save_path.with_suffix(".png"))


def make_radar_chart(
    data_list: list[dict[str, float]], labels: list[str], save_path: Path
) -> None:
    # Number of variables
    num_vars = len(data_list[0])

    # Compute angle each axis will occupy
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # Create figure and polar subplot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Set y-axis limits to always be between 0 and 1
    ax.set_ylim([0, 1])

    # Assign a unique color to each chart
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    # Loop through each data dictionary and plot on the same subplot
    for i, data in enumerate(data_list):
        values = list(data.values())
        values += values[:1]
        ax.plot(angles, values, color=colors[i], label=labels[i])

    # Draw labels for the axes
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(data_list[0].keys()))

    # Add a legend
    ax.legend()

    # Save the plot as a PNG file
    plt.savefig(save_path.with_suffix(".png"))
