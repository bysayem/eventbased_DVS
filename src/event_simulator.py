from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def generate_moving_square(
    width=160,
    height=120,
    frame_count=80,
    square_size=20,
    start_x=10,
    start_y=50,
    speed_x=2,
    threshold=0.15,
):
    """Generate synthetic DVS events from a moving square."""

    previous_frame = np.zeros((height, width), dtype=np.float32)
    events = []
    frames = []

    for frame_index in range(frame_count):
        current_frame = np.zeros((height, width), dtype=np.float32)

        x_position = min(
            start_x + frame_index * speed_x,
            width - square_size,
        )

        x_end = x_position + square_size
        y_end = start_y + square_size

        current_frame[start_y:y_end, x_position:x_end] = 1.0
        frames.append(current_frame.copy())

        difference = current_frame - previous_frame
        timestamp = frame_index / 30.0

        positive_y, positive_x = np.where(difference > threshold)
        negative_y, negative_x = np.where(difference < -threshold)

        for x, y in zip(positive_x, positive_y):
            events.append([x, y, timestamp, 1])

        for x, y in zip(negative_x, negative_y):
            events.append([x, y, timestamp, -1])

        previous_frame = current_frame

    return np.asarray(events, dtype=np.float32), frames


def save_events_as_csv(events, output_path):
    """Save events in a human-readable CSV format."""

    header = "x,y,timestamp,polarity"
    np.savetxt(
        output_path,
        events,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%d", "%.6f", "%d"],
    )


def plot_event_space(events, output_path, width, height):
    """Plot event locations in image coordinates."""

    positive = events[events[:, 3] == 1]
    negative = events[events[:, 3] == -1]

    plt.figure(figsize=(8, 6))

    if len(positive) > 0:
        plt.scatter(
            positive[:, 0],
            positive[:, 1],
            s=8,
            c="red",
            alpha=0.8,
            label="Positive",
        )

    if len(negative) > 0:
        plt.scatter(
            negative[:, 0],
            negative[:, 1],
            s=8,
            c="blue",
            alpha=0.8,
            label="Negative",
        )

    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.title("DVS Events in Image Coordinates")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_event_timeline(events, output_path):
    """Plot event locations over time."""

    positive = events[events[:, 3] == 1]
    negative = events[events[:, 3] == -1]

    plt.figure(figsize=(10, 5))

    if len(positive) > 0:
        plt.scatter(
            positive[:, 2],
            positive[:, 0],
            s=5,
            c="red",
            alpha=0.7,
            label="Positive",
        )

    if len(negative) > 0:
        plt.scatter(
            negative[:, 2],
            negative[:, 0],
            s=5,
            c="blue",
            alpha=0.7,
            label="Negative",
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Pixel X")
    plt.title("DVS Event Activity Over Time")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_example_frame(frame, output_path):
    """Save one ordinary frame for visual reference."""

    plt.figure(figsize=(7, 5))
    plt.imshow(frame, cmap="gray", vmin=0, vmax=1)
    plt.title("Example Moving Square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events, frames = generate_moving_square()

    np.save(OUTPUT_DIR / "events.npy", events)

    save_events_as_csv(
        events,
        OUTPUT_DIR / "events.csv",
    )

    plot_event_space(
        events,
        FIGURES_DIR / "event_xy.png",
        width=160,
        height=120,
    )

    plot_event_timeline(
        events,
        FIGURES_DIR / "event_timeline.png",
    )

    plot_example_frame(
        frames[20],
        FIGURES_DIR / "example_frame.png",
    )

    positive_count = np.sum(events[:, 3] == 1)
    negative_count = np.sum(events[:, 3] == -1)

    np.save(OUTPUT_DIR / "frames.npy", np.asarray(frames))
    np.save(OUTPUT_DIR / "events.npy", events)
    np.save(OUTPUT_DIR / "frames.npy", np.asarray(frames))

    print(f"Generated events: {len(events)}")
    print(f"Positive events: {positive_count}")
    print(f"Negative events: {negative_count}")
    print(f"Data shape: {events.shape}")
    print("Columns: x, y, timestamp, polarity")
    print("Saved: outputs/events.npy")
    print("Saved: outputs/events.csv")
    print("Saved: outputs/figures/event_xy.png")
    print("Saved: outputs/figures/event_timeline.png")
    print("Saved: outputs/figures/example_frame.png")


if __name__ == "__main__":
    main()