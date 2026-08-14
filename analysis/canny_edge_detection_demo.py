"""Run and visualise each stage of the Canny edge detector.

Replace ``make_demo_map()`` with a Brillouin 2D map to tune the thresholds on
experimental data. The detector itself lives in ``feature_extraction.py``.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_extraction import canny_edges, plot_canny_stages



image = plt.imread("p.png")


if image.ndim == 3:
    image = image[..., :3]  # remove alpha channel if present
    image = (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    )

edges, stages = canny_edges(
    image,
    sigma=1.5,
    low_threshold=0.08,
    high_threshold=0.20,
    return_stages=True,
)

plot_canny_stages(stages)
plt.show()