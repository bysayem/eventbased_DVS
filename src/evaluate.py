from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.npy"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "tracking_results.npy"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metrics.txt"

WIDTH = 160
HEIGHT = 120
FRAME_RATE = 30.0
FRAME_COUNT = 80
SQUARE_SIZE = 20
START_X = 10
SPEED_X = 2


def ground_truth_center_x(timestamps):
    """Return the true square center for each timestamp."""

    frame_indices = np.rint(timestamps * FRAME_RATE).astype(int)
    frame_indices = np.clip(frame_indices, 0, FRAME_COUNT - 1)

    left_edges = START_X + frame_indices * SPEED_X
    left_edges = np.minimum(left_edges, WIDTH - SQUARE_SIZE)

    return left_edges + SQUARE_SIZE / 2


def calculate_metrics(events, tracking_results):
    """Calculate tracking and event-sparsity metrics."""

    valid = ~np.isnan(tracking_results[:, 1])

    if not np.any(valid):
        raise RuntimeError("No valid tracking points found.")

    timestamps = tracking_results[valid, 0]
    estimated_x = tracking_results[valid, 1]

    true_x = ground_truth_center_x(timestamps)
    absolute_errors = np.abs(estimated_x - true_x)

    mae = np.mean(absolute_errors)
    max_error = np.max(absolute_errors)
    valid_rate = np.mean(valid) * 100.0

    total_possible_frame_pixels = (
        FRAME_COUNT * WIDTH * HEIGHT
    )
    event_count = len(events)

    event_density = (
        event_count / total_possible_frame_pixels
    ) * 100.0

    data_reduction = (
        1.0 - event_count / total_possible_frame_pixels
    ) * 100.0

    return {
        "tracking_mae_pixels": mae,
        "maximum_error_pixels": max_error,
        "valid_tracking_rate_percent": valid_rate,
        "total_events": event_count,
        "event_density_percent": event_density,
        "estimated_data_reduction_percent": data_reduction,
    }, timestamps, true_x, estimated_x


def plot_tracking_error(
    timestamps,
    true_x,
    estimated_x,
    output_path,
):
    """Plot ground truth, estimate and error."""

    errors = np.abs(estimated_x - true_x)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    axes[0].plot(
        timestamps,
        true_x,
        label="Ground truth",
        linewidth=2,
    )

    axes[0].plot(
        timestamps,
        estimated_x,
        "--",
        label="Event-based estimate",
        linewidth=2,
    )

    axes[0].set_ylabel("Center X (pixels)")
    axes[0].set_title("Event-Based Tracking Evaluation")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        timestamps,
        errors,
        color="darkorange",
        linewidth=2,
    )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Absolute error (pixels)")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events = np.load(EVENTS_PATH)
    tracking_results = np.load(RESULTS_PATH)

    metrics, timestamps, true_x, estimated_x = (
        calculate_metrics(
            events,
            tracking_results,
        )
    )

    plot_tracking_error(
        timestamps,
        true_x,
        estimated_x,
        FIGURES_DIR / "tracking_evaluation.png",
    )

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write("Neuromorphic Event-Based Vision Metrics\n")
        file.write("=" * 45 + "\n")

        for name, value in metrics.items():
            file.write(f"{name}: {value:.4f}\n")

    print("Evaluation completed.")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Saved: outputs/metrics.txt")
    print(
        "Saved: "
        "outputs/figures/tracking_evaluation.png"
    )


if __name__ == "__main__":
    main()