from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.npy"
TRACKING_PATH = PROJECT_ROOT / "outputs" / "tracking_results.npy"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "event_tracking.gif"


def main():
    events = np.load(EVENTS_PATH)
    tracking = np.load(TRACKING_PATH)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    event_axis, trajectory_axis = axes

    event_axis.set_xlim(0, 160)
    event_axis.set_ylim(120, 0)
    event_axis.set_title("DVS Event Stream")
    event_axis.set_xlabel("Pixel X")
    event_axis.set_ylabel("Pixel Y")

    trajectory_axis.set_xlim(0, 160)
    trajectory_axis.set_ylim(120, 0)
    trajectory_axis.set_title("Tracked Trajectory")
    trajectory_axis.set_xlabel("Pixel X")
    trajectory_axis.set_ylabel("Pixel Y")

    positive = events[events[:, 3] == 1]
    negative = events[events[:, 3] == -1]

    positive_plot = event_axis.scatter(
        [], [], s=8, c="red", label="Positive"
    )

    negative_plot = event_axis.scatter(
        [], [], s=8, c="blue", label="Negative"
    )

    trajectory_line, = trajectory_axis.plot(
        [],
        [],
        "g-",
        linewidth=2,
        label="Estimated path",
    )

    current_position, = trajectory_axis.plot(
        [],
        [],
        "ko",
        markersize=7,
    )

    event_axis.legend(loc="upper right")

    def update(frame_index):
        current_time = tracking[frame_index, 0]

        positive_current = positive[
            positive[:, 2] <= current_time
        ]

        negative_current = negative[
            negative[:, 2] <= current_time
        ]

        positive_plot.set_offsets(
            positive_current[:, :2]
            if len(positive_current) > 0
            else np.empty((0, 2))
        )

        negative_plot.set_offsets(
            negative_current[:, :2]
            if len(negative_current) > 0
            else np.empty((0, 2))
        )

        valid = ~np.isnan(tracking[: frame_index + 1, 1])

        trajectory_line.set_data(
            tracking[: frame_index + 1, 1][valid],
            tracking[: frame_index + 1, 2][valid],
        )

        if valid.any():
            current_position.set_data(
                [tracking[frame_index, 1]],
                [tracking[frame_index, 2]],
            )

        figure.suptitle(
            f"Event-Based Motion Tracking | "
            f"t={current_time:.2f}s"
        )

        return (
            positive_plot,
            negative_plot,
            trajectory_line,
            current_position,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(tracking),
        interval=80,
        blit=False,
    )

    animation.save(
        OUTPUT_PATH,
        writer="pillow",
        fps=12,
    )

    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()