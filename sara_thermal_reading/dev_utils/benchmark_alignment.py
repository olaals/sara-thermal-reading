import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sara_thermal_reading.config.settings import settings
from sara_thermal_reading.file_io.blob import BlobStore
from sara_thermal_reading.main_thermal_workflow import process_thermal_image


def run_benchmark_alignment(
    csv_path: str,
    src_st_acc: str,
    ref_st_acc: str,
    installation_code: str,
    results_path: str,
) -> None:
    """Interactively evaluate image alignment quality and save verdicts to JSON."""
    threshold = settings.MIN_ALIGNMENT_SCORE

    # Read folder paths from file
    folders = Path(csv_path).read_text().strip().splitlines()
    folders = [f.strip() for f in folders if f.strip()]

    source_blob_store = BlobStore.from_account_name(installation_code, src_st_acc)
    reference_blob_store = BlobStore.from_account_name(installation_code, ref_st_acc)

    # Load existing results if the file exists (allows resuming)
    results: list[dict] = []
    evaluated_blobs: set[str] = set()
    if Path(results_path).exists():
        existing = json.loads(Path(results_path).read_text())
        results = existing
        evaluated_blobs = {r["blob_name"] for r in results}
        print(f"Loaded {len(results)} existing results from {results_path}")

    for folder in folders:
        print(f"\nListing blobs in folder: {folder}")
        all_blobs = source_blob_store.list_blobs_by_prefix(prefix=folder + "/")
        tiff_blobs = [
            b for b in all_blobs if b.endswith(".tiff") and "__ThermalImage__" in b
        ]

        if not tiff_blobs:
            print(f"  No thermal tiff files found in {folder}, skipping.")
            continue

        # Group by tag_id + description to avoid re-downloading references
        ref_cache: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}

        for blob_name in tiff_blobs:
            if blob_name in evaluated_blobs:
                print(f"  Already evaluated: {blob_name.split('/')[-1]}, skipping.")
                continue

            filename = blob_name.split("/")[-1]
            parts = filename.split("__")
            tag_id = parts[0]
            inspection_description = parts[2].replace("-", " ")
            ref_key = f"{tag_id}_{inspection_description}"

            # Download reference if not cached
            if ref_key not in ref_cache:
                ref_image_blob = f"{ref_key}/{settings.REFERENCE_IMAGE_TIFF_FILENAME}"
                ref_polygon_blob = f"{ref_key}/{settings.REFERENCE_POLYGON_FILENAME}"
                try:
                    ref_image = reference_blob_store.download_thermal_tiff(
                        ref_image_blob
                    )
                    ref_polygon = reference_blob_store.download_polygon(
                        ref_polygon_blob
                    )
                    ref_cache[ref_key] = (ref_image, ref_polygon)
                except Exception as e:
                    print(f"  Could not download reference for {ref_key}: {e}")
                    continue

            reference_image, reference_polygon = ref_cache[ref_key]

            # Download source image
            try:
                source_image = source_blob_store.download_thermal_tiff(blob_name)
            except Exception as e:
                print(f"  Could not download source {blob_name}: {e}")
                continue

            # Run alignment
            try:
                (
                    _temperature,
                    _annotated_image,
                    warped_polygon_list,
                    _warped_ref_img,
                    alignment_score,
                ) = process_thermal_image(
                    reference_image, source_image, reference_polygon
                )
            except Exception as e:
                print(f"  Error processing {filename}: {e}, skipping.")
                continue

            # Interactive display
            user_verdict = _show_benchmark_match(
                reference_image=reference_image,
                source_image=source_image,
                reference_polygon=reference_polygon,
                warped_polygon_list=warped_polygon_list,
                alignment_score=alignment_score,
                threshold=threshold,
                blob_name=blob_name,
            )

            if user_verdict is None:
                print("  Skipped (window closed without input).")
                continue

            results.append(
                {
                    "blob_name": blob_name,
                    "alignment_score": alignment_score,
                    "verdict": user_verdict,
                }
            )
            verdict_str = "GOOD" if user_verdict else "BAD"
            print(f"  {filename} | score={alignment_score:.4f} | verdict={verdict_str}")

            # Save after each verdict so progress is not lost
            Path(results_path).write_text(json.dumps(results, indent=2))

    print(f"\nDone. {len(results)} results saved to {results_path}")


def generate_benchmark_plots(
    results_path: str,
    output_path: str,
    line_plot_path: str,
    threshold: float | None = None,
) -> None:
    """Generate a pie chart and threshold sweep plot from benchmark results."""
    if threshold is None:
        threshold = settings.MIN_ALIGNMENT_SCORE

    results_file = Path(results_path)
    if not results_file.exists():
        print(f"Results file not found: {results_path}")
        return

    results = json.loads(results_file.read_text())
    if not results:
        print("No results in file. Exiting.")
        return

    # --- Confusion matrix at the specified threshold ---
    true_positive = sum(
        1 for r in results if r["verdict"] and r["alignment_score"] >= threshold
    )
    false_negative = sum(
        1 for r in results if r["verdict"] and r["alignment_score"] < threshold
    )
    false_positive = sum(
        1 for r in results if not r["verdict"] and r["alignment_score"] >= threshold
    )
    true_negative = sum(
        1 for r in results if not r["verdict"] and r["alignment_score"] < threshold
    )

    cm = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    cm_labels = np.array([["TP", "FN"], ["FP", "TN"]])
    colors_map = np.array([["#4CAF50", "#FFC107"], ["#F44336", "#2196F3"]])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

    for i in range(2):
        for j in range(2):
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor=colors_map[i, j],
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=2,
                )
            )
            ax.text(
                j,
                i,
                f"{cm_labels[i, j]}\n{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        [f"Positive\n(score >= {threshold})", f"Negative\n(score < {threshold})"],
        fontsize=11,
    )
    ax.set_yticklabels(
        ["Actual Positive", "Actual Negative"],
        fontsize=11,
    )
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Alignment Benchmark — Confusion Matrix\n"
        f"Threshold: {threshold} | Total: {len(results)} images",
        fontsize=13,
    )
    ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to: {output_path}")

    print(
        f"Results at threshold {threshold}: TP={true_positive}, FN={false_negative}, "
        f"FP={false_positive}, TN={true_negative}"
    )

    # --- Threshold sweep line plot ---
    scores = [r["alignment_score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    sweep_thresholds = np.linspace(min_score, max_score, 200)

    tp_counts = []
    fn_counts = []
    fp_counts = []
    tn_counts = []

    for t in sweep_thresholds:
        tp_counts.append(
            sum(1 for r in results if r["verdict"] and r["alignment_score"] >= t)
        )
        fn_counts.append(
            sum(1 for r in results if r["verdict"] and r["alignment_score"] < t)
        )
        fp_counts.append(
            sum(1 for r in results if not r["verdict"] and r["alignment_score"] >= t)
        )
        tn_counts.append(
            sum(1 for r in results if not r["verdict"] and r["alignment_score"] < t)
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sweep_thresholds,
        tp_counts,
        color="#4CAF50",
        linewidth=2,
        label="Correct match, above threshold (TP)",
    )
    ax.plot(
        sweep_thresholds,
        fn_counts,
        color="#FFC107",
        linewidth=2,
        label="Correct match, below threshold (FN)",
    )
    ax.plot(
        sweep_thresholds,
        fp_counts,
        color="#F44336",
        linewidth=2,
        label="Wrong match, above threshold (FP)",
    )
    ax.plot(
        sweep_thresholds,
        tn_counts,
        color="#2196F3",
        linewidth=2,
        label="Wrong match, below threshold (TN)",
    )

    ax.axvline(
        x=threshold,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Current threshold ({threshold})",
    )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Number of cases")
    ax.set_title("Alignment Benchmark: Threshold Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(line_plot_path, dpi=150)
    plt.close(fig)
    print(f"Threshold sweep plot saved to: {line_plot_path}")


def _show_benchmark_match(
    reference_image: np.ndarray,
    source_image: np.ndarray,
    reference_polygon: list[tuple[int, int]],
    warped_polygon_list: list[tuple[int, int]],
    alignment_score: float,
    threshold: float,
    blob_name: str,
) -> bool | None:
    """Show alignment result and wait for y/n keypress. Returns True/False/None."""
    polygon_np = np.array(reference_polygon)
    warped_polygon_np = np.array(warped_polygon_list)

    result: dict[str, bool | None] = {"verdict": None}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Score: {alignment_score:.4f} | Threshold: {threshold:.4f} | "
        f"{'ABOVE' if alignment_score >= threshold else 'BELOW'}\n"
        f"{blob_name}\n"
        f"Press Y (good match) or N (bad match)",
        fontsize=10,
    )

    ax1.set_title("Reference image")
    ax1.imshow(reference_image, cmap="jet")
    ax1.fill(
        polygon_np[:, 0],
        polygon_np[:, 1],
        facecolor="red",
        edgecolor="white",
        linewidth=2,
        alpha=0.5,
    )
    ax1.axis("off")

    ax2.set_title("Source image (with warped polygon)")
    ax2.imshow(source_image, cmap="jet")
    ax2.fill(
        warped_polygon_np[:, 0],
        warped_polygon_np[:, 1],
        facecolor="red",
        edgecolor="white",
        linewidth=2,
        alpha=0.5,
    )
    ax2.axis("off")

    def on_key(event: Any) -> None:
        if event.key == "y":
            result["verdict"] = True
            plt.close(fig)
        elif event.key == "n":
            result["verdict"] = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    return result["verdict"]
