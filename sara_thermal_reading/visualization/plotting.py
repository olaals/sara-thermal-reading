import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from sara_thermal_reading.file_io.tiff_loader import load_thermal_tiff_from_bytes


def plot_thermal_image(
    image: np.ndarray,
    title: str,
    polygon_points: Optional[list[tuple[int, int]]] = None,
    ax: Optional[Axes] = None,
) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(image, cmap="jet")
    plt.colorbar(im, ax=ax, label="Temperature")
    ax.set_title(title)

    if polygon_points:
        points = np.array(polygon_points)
        points = np.vstack(
            [points, points[0]]
        )  # Closing the polygon (last->first point)
        ax.plot(points[:, 0], points[:, 1], color="lime", linestyle="-", linewidth=2)


def plot_thermal_from_path(
    file_path: Path, polygon_json_path: Optional[Path] = None
) -> None:
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    image = load_thermal_tiff_from_bytes(file_bytes)

    polygon_points = None
    if polygon_json_path:
        with open(polygon_json_path, "r") as f:
            polygon_points = json.load(f)

    plot_thermal_image(image, f"Thermal Image: {file_path}", polygon_points)
    plt.show()
