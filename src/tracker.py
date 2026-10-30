from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.npy"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

WIDTH = 160
HEIGHT = 120
TAU = 0.08
ACTIVITY_THRESHOLD = 0.15


def build_latest_timestamp_maps(events, current_time, width, height):
    """Create latest positive and negative event timestamp maps."""

    positive_latest = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32,
    )
    negative_latest = np.full(
        (height, width),
        -np.inf,
        dtype=np.float32,
    )

    past_events = events[events[:, 2] <= current_time]

    for x_value, y_value, timestamp, polarity in past_events:
        x = int(x_value)
        y = int(y_value)

        if polarity == 1:
            positive_latest[y, x] = max(
                positive_latest[y, x],
                timestamp,
            )
        else:
            negative_latest[y, x] = max(
                negative_latest[y, x],
                timestamp,
            )

    return positive_latest, negative_latest


def build_combined_time_surface(events, current_time):
    """Build a combined RGB time surface."""

    positive_latest, negative_latest = (
        build_latest_timestamp_maps(
            events,
            current_time,
            WIDTH,
            HEIGHT,
        )
    )

    positive_surface = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    negative_surface = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    positive_mask = np.isfinite(positive_latest)
    negative_mask = np.isfinite(negative_latest)

    positive_surface[positive_mask] = np.exp(
        -(current_time - positive_latest[positive_mask]) / TAU
    )

    negative_surface[negative_mask] = np.exp(
        -(current_time - negative_latest[negative_mask]) / TAU
    )

    combined_surface = np.maximum(
        positive_surface,
        negative_surface,
    )

    return combined_surface


def estimate_object_center(surface):
    """Estimate object center from active time-surface pixels."""

    active_y, active_x = np.where(
        surface >= ACTIVITY_THRESHOLD
    )

    if len(active_x) == 0:
        return np.nan, np.nan, 0

    weights = surface[active_y, active_x]

    center_x = np.average(active_x, weights=weights)
    center_y = np.average(active_y, weights=weights)

    return center_x, center_y, len(active_x)


def track_object(events):
    """Estimate object position over time."""

    timestamps = np.linspace(
        events[:, 2].min(),
        events[:, 2].max(),
        num=80,
    )

    tracking_results = []

    for current_time in timestamps:
        surface = build_combined_time_surface(
            events,
            current_time,
        )

        center_x, center_y, active_pixels = (
            estimate_object_center(surface)
        )

        tracking_results.append(
            [
                current_time,
                center_x,
                center_y,
                active_pixels,
            ]
        )

    return np.asarray(tracking_results)


def plot_trajectory(results, output_path):
    """Plot estimated object trajectory."""

    valid = ~np.isnan(results[:, 1])

    time = results[valid, 0]
    estimated_x = results[valid, 1]

    # Ground-truth center from the simulator.
    ground_truth_x = 20 + time * 30 * 2

    ground_truth_x = np.minimum(
        ground_truth_x,
        WIDTH - 10,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        time,
        ground_truth_x,
        label="Ground-truth X",
        linewidth=2,
    )

    plt.plot(
        time,
        estimated_x,
        "--",
        label="Estimated X",
        linewidth=2,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Object center X (pixels)")
    plt.title("Event-Based Object Tracking")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trajectory_path(results, output_path):
    """Plot the estimated spatial trajectory."""

    valid = ~np.isnan(results[:, 1])

    plt.figure(figsize=(7, 5))

    plt.plot(
        results[valid, 1],
        results[valid, 2],
        "o-",
        markersize=3,
        label="Estimated trajectory",
    )

    plt.xlim(0, WIDTH)
    plt.ylim(HEIGHT, 0)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.title("Tracked Object Path")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events = np.load(EVENTS_PATH)
    results = track_object(events)

    np.save(
        PROJECT_ROOT / "outputs" / "tracking_results.npy",
        results,
    )

    plot_trajectory(
        results,
        FIGURES_DIR / "tracking_x_over_time.png",
    )

    plot_trajectory_path(
        results,
        FIGURES_DIR / "tracking_path.png",
    )

    valid = ~np.isnan(results[:, 1])

    print(f"Loaded events: {len(events)}")
    print(f"Tracking points: {len(results)}")
    print(f"Valid tracking points: {np.sum(valid)}")
    print("Saved: outputs/tracking_results.npy")
    print(
        "Saved: "
        "outputs/figures/tracking_x_over_time.png"
    )
    print(
        "Saved: "
        "outputs/figures/tracking_path.png"
    )


if __name__ == "__main__":
    main()