from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.npy"
FRAMES_PATH = PROJECT_ROOT / "outputs" / "frames.npy"
TRACKING_PATH = PROJECT_ROOT / "outputs" / "tracking_results.npy"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "baseline_comparison.txt"


def detect_frame_centroid(frame):
    """Detect object center from one grayscale frame."""

    object_y, object_x = np.where(frame > 0.5)

    if len(object_x) == 0:
        return np.nan, np.nan

    return np.mean(object_x), np.mean(object_y)


def track_with_frames(frames):
    """Track the square using complete image frames."""

    results = []

    for frame_index, frame in enumerate(frames):
        center_x, center_y = detect_frame_centroid(frame)
        timestamp = frame_index / 30.0

        results.append(
            [timestamp, center_x, center_y]
        )

    return np.asarray(results)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events = np.load(EVENTS_PATH)
    frames = np.load(FRAMES_PATH)
    event_tracking = np.load(TRACKING_PATH)

    frame_tracking = track_with_frames(frames)

    event_valid = ~np.isnan(event_tracking[:, 1])
    frame_valid = ~np.isnan(frame_tracking[:, 1])

    event_time = event_tracking[event_valid, 0]
    event_x = event_tracking[event_valid, 1]

    frame_time = frame_tracking[frame_valid, 0]
    frame_x = frame_tracking[frame_valid, 1]

    frame_x_interpolated = np.interp(
        event_time,
        frame_time,
        frame_x,
    )

    event_errors = np.abs(event_x - frame_x_interpolated)
    event_mae = np.mean(event_errors)

    total_frames = len(frames)
    height, width = frames[0].shape
    full_frame_pixels = total_frames * height * width
    total_events = len(events)

    event_density = (
        total_events / full_frame_pixels
    ) * 100.0

    data_reduction = (
        1.0 - total_events / full_frame_pixels
    ) * 100.0

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )

    axes[0].plot(
        event_time,
        frame_x_interpolated,
        label="Frame-based centroid",
        linewidth=2,
    )

    axes[0].plot(
        event_time,
        event_x,
        "--",
        label="Event-based tracker",
        linewidth=2,
    )

    axes[0].set_ylabel("Center X (pixels)")
    axes[0].set_title(
        "Event-Based vs Frame-Based Tracking"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        event_time,
        event_errors,
        color="darkorange",
        linewidth=2,
    )

    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Absolute error (pixels)")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "baseline_comparison.png",
        dpi=150,
    )
    plt.close(figure)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("Event-Based vs Frame-Based Comparison\n")
        file.write("=" * 45 + "\n")
        file.write(
            f"Event-based MAE vs frame baseline: "
            f"{event_mae:.4f} pixels\n"
        )
        file.write(
            f"Total events processed: {total_events}\n"
        )
        file.write(
            f"Full-frame pixels processed: "
            f"{full_frame_pixels}\n"
        )
        file.write(
            f"Event density: {event_density:.4f}%\n"
        )
        file.write(
            f"Estimated data reduction: "
            f"{data_reduction:.4f}%\n"
        )

    print("Actual baseline comparison completed.")
    print(
        f"Event-based MAE vs frame baseline: "
        f"{event_mae:.4f} pixels"
    )
    print(f"Total events: {total_events}")
    print(f"Full-frame pixels: {full_frame_pixels}")
    print(f"Event density: {event_density:.4f}%")
    print(
        f"Estimated data reduction: "
        f"{data_reduction:.4f}%"
    )
    print(
        "Saved: outputs/figures/baseline_comparison.png"
    )
    print("Saved: outputs/baseline_comparison.txt")


if __name__ == "__main__":
    main()