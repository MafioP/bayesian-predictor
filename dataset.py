"""
Synthetic vision dataset for distance regression.

Each image shows one fixed, irregular (asymmetric) polygon -- "the object"
-- placed with a random position, rotation, and scale. Scale encodes
distance: closer objects look bigger, same as monocular size cues in real
images. Using the *same* base shape every time, rather than a new random
shape per sample, means the model has to learn to recognize this one
object under arbitrary 2D transforms -- closer to real single-object
distance estimation, and a harder problem for convolution specifically,
since CNNs are translation-invariant but not rotation-invariant by
construction.

The true physical projection (apparent size proportional to 1/distance)
would shrink far-away objects to sub-pixel size well before MAX_DISTANCE,
so we use a gentler 1/sqrt(distance) falloff instead -- a deliberate
simplification to keep the whole distance range visually learnable rather
than physically exact.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

IMAGE_SIZE = 224
MIN_DISTANCE = 1.0
MAX_DISTANCE = 500.0

# radius = SIZE_SCALE / sqrt(distance), clamped to sane pixel bounds.
# Calibrated so radius == MAX_RADIUS at MIN_DISTANCE and shrinks to
# ~MIN_RADIUS only in the last ~20% of the range (see module docstring).
SIZE_SCALE = 60.0
MIN_RADIUS = 3
MAX_RADIUS = 60

# One fixed, irregular (asymmetric) base shape, given as (x, y) offsets
# from its own center at roughly unit scale. Every sample draws a scaled,
# rotated, translated copy of *this same shape* -- scale, rotation, and
# position are the only things that vary from sample to sample.
BASE_SHAPE = [
    (1.0, 0.0),
    (0.3, 0.4),
    (0.5, 1.0),
    (-0.2, 0.6),
    (-1.0, 0.2),
    (-0.6, -0.5),
    (0.0, -1.0),
    (0.6, -0.4),
]
_BASE_SHAPE_MAX_NORM = max(np.hypot(x, y) for x, y in BASE_SHAPE)


def _distance_to_radius(distance: float) -> float:
    radius = SIZE_SCALE / np.sqrt(distance)
    return float(np.clip(radius, MIN_RADIUS, MAX_RADIUS))


def _transform_base_shape(cx: float, cy: float, radius: float, angle: float):
    """Scale BASE_SHAPE by radius, rotate by angle (radians), translate to (cx, cy)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    points = []
    for x, y in BASE_SHAPE:
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        points.append((cx + radius * rx, cy + radius * ry))
    return points


class DistanceDataset(Dataset):
    """Generates (image, distance) pairs on the fly."""

    def __init__(self, num_samples: int, seed: int = 0, noise_std: float = 0.05):
        self.num_samples = num_samples
        self.noise_std = noise_std
        # Each sample gets its own deterministic seed so train/val splits
        # drawn from different seeds never overlap.
        self.rng = np.random.default_rng(seed)
        self.distances = self.rng.uniform(MIN_DISTANCE, MAX_DISTANCE, size=num_samples)
        self.seeds = self.rng.integers(0, 2**31 - 1, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        distance = self.distances[idx]
        rng = np.random.default_rng(self.seeds[idx])

        radius = _distance_to_radius(distance)
        # Rotation preserves each vertex's distance from center, so the
        # margin only needs the base shape's max extent, not the angle.
        margin = int(np.ceil(radius * _BASE_SHAPE_MAX_NORM)) + 1
        cx = rng.integers(margin, IMAGE_SIZE - margin)
        cy = rng.integers(margin, IMAGE_SIZE - margin)
        angle = rng.uniform(0, 2 * np.pi)

        noise = rng.normal(loc=0.2, scale=self.noise_std, size=(IMAGE_SIZE, IMAGE_SIZE))
        noise = np.clip(noise, 0.0, 1.0)
        image = Image.fromarray((noise * 255).astype(np.uint8), mode="L")

        draw = ImageDraw.Draw(image)
        points = _transform_base_shape(cx, cy, radius, angle)
        draw.polygon(points, fill=230)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image).unsqueeze(0)  # (1, H, W) grayscale
        distance_t = torch.tensor(distance, dtype=torch.float32)
        return image_t, distance_t
