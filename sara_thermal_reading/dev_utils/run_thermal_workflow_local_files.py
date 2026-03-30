import json
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sara_thermal_reading.dev_utils.image_warping import random_translate_thermal_img
from sara_thermal_reading.file_io.tiff_loader import load_thermal_tiff
from sara_thermal_reading.main_thermal_workflow import process_thermal_image
from sara_thermal_reading.visualization.plotting import plot_thermal_image


def run_thermal_workflow_local_files(
    polygon_path: str,
    reference_image_path: str,
    source_image_path: Optional[str] = None,
    tag_id: str = "test_tag_id",
    inspection_description: str = "test_inspection_description",
) -> None:
    """
    Runs the FFF workflow using local files.
    If source_image_path is not provided, generates a warped source image from the reference.
    """
    reference_image = load_thermal_tiff(reference_image_path)

    with open(polygon_path, "r") as f:
        polygon_points = json.load(f)

    # Use source image if provided, else generate a "mock" source image
    # by warping the reference image
    if source_image_path:
        source_image = load_thermal_tiff(source_image_path)
        source_title = "Source Image (Loaded)"
    else:
        print("No source image provided. Generating warped image from reference...")
        source_image = random_translate_thermal_img(reference_image)
        source_title = "Source Image (Generated/Warped)"

    (
        temperature,
        annotated_image,
        warped_polygon,
        warped_reference_img,
    ) = process_thermal_image(reference_image, source_image, polygon_points)

    print(f"Workflow complete. Temperature: {temperature:.2f}°C")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_thermal_image(
        reference_image,
        "Reference Image",
        polygon_points=polygon_points,
        ax=axes[0],
    )

    plot_thermal_image(
        source_image,
        source_title,
        ax=axes[1],
    )

    axes[2].imshow(warped_reference_img, cmap="gray")
    axes[2].set_title("Warped Reference (Aligned to Source)")

    warped_poly_pts = np.array(warped_polygon)
    warped_poly_pts = np.vstack([warped_poly_pts, warped_poly_pts[0]])
    axes[2].plot(
        warped_poly_pts[:, 0],
        warped_poly_pts[:, 1],
        color="lime",
        linestyle="-",
        linewidth=2,
    )

    axes[3].imshow(annotated_image)
    axes[3].set_title(f"Result (Temperature: {temperature:.2f})")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
