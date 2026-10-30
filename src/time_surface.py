from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

WIDTH = 160
HEIGHT = 120
TAU = 0.08


def build_time_surface(events, current_time, width, height, tau):
    """
    Build positive and negative exponential time surfaces.

    Event format:
        [x, y, timestamp, polarity]
    """

    positive_surface = np.zeros((height, width), dtype=np.float32)
    negative_surface = np.zeros((height, width), dtype=np.float32)

    past_events = events[events[:, 2] <= current_time]

    if len(past_events) == 0:
        return positive_surface, negative_surface

    for polarity, surface in [(1, positive_surface), (-1, negative_surface)]:
        polarity_events = past_events[past_events[:, 3] == polarity]

        if len(polarity_events) == 0:
            continue

        for y in range(height):
            row_events = polarity_events[polarity_events[:, 1] == y]

            if len(row_events) == 0:
                continue

            for x in range(width):
                pixel_events = row_events[row_events[:, 0] == x]

                if len(pixel_events) == 0:
                    continue

                latest_timestamp = pixel_events[-1, 2]
                time_difference = current_time - latest_timestamp
                surface[y, x] = np.exp(-time_difference / tau)

    return positive_surface, negative_surface


def create_color_time_surface(positive_surface, negative_surface):
    """
    Combine positive and negative surfaces into an RGB image.

    Positive events appear red.
    Negative events appear blue.
    """

    image = np.zeros(
        (positive_surface.shape[0], positive_surface.shape[1], 3),
        dtype=np.float32,
    )

    image[:, :, 0] = positive_surface
    image[:, :, 2] = negative_surface

    return image


def save_time_surface_figure(
    positive_surface,
    negative_surface,
    current_time,
    output_path,
):
    """Save positive, negative and combined time surfaces."""

    color_surface = create_color_time_surface(
        positive_surface,
        negative_surface,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(positive_surface, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Positive Time Surface")
    axes[0].axis("off")

    axes[1].imshow(negative_surface, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Negative Time Surface")
    axes[1].axis("off")

    axes[2].imshow(color_surface)
    axes[2].set_title(f"Combined Surface: t={current_time:.2f}s")
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    events = np.load(EVENTS_PATH)

    sample_times = np.linspace(
        events[:, 2].min(),
        events[:, 2].max(),
        num=6,
    )

    time_surfaces = []

    for index, current_time in enumerate(sample_times):
        positive_surface, negative_surface = build_time_surface(
            events=events,
            current_time=current_time,
            width=WIDTH,
            height=HEIGHT,
            tau=TAU,
        )

        combined_surface = create_color_time_surface(
            positive_surface,
            negative_surface,
        )

        time_surfaces.append(combined_surface)

        output_path = FIGURES_DIR / f"time_surface_{index:02d}.png"

        save_time_surface_figure(
            positive_surface=positive_surface,
            negative_surface=negative_surface,
            current_time=current_time,
            output_path=output_path,
        )

    time_surfaces = np.asarray(time_surfaces)

    np.save(
        OUTPUT_DIR / "time_surfaces.npy",
        time_surfaces,
    )

    print(f"Loaded events: {len(events)}")
    print(f"Time constant tau: {TAU} seconds")
    print(f"Generated time surfaces: {len(time_surfaces)}")
    print("Saved: outputs/time_surfaces.npy")

    for index in range(len(sample_times)):
        print(f"Saved: outputs/figures/time_surface_{index:02d}.png")


if __name__ == "__main__":
    main()