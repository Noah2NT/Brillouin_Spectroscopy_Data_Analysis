"""Known-truth, continuously varying spectral cubes for reconstruction studies.

Uses the simulator's line generation, instrument convolution and detector.
These are custom Brillouin-window scenes, not vendor-format full VIPA scans.
The same clean rate cube is held fixed across exposures and noise realizations.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

from simulator.core.instrument import Instrument
from simulator.core.materials import Material
from simulator.core.spectrum import compose_densities
from simulator.core.noise import apply_detector


SCENES = ("hydrogel_gradient", "cell_domains", "fine_structure")


def make_scene(name, size=28, bins=512):
    if name not in SCENES or size < 12 or bins < 32:
        raise ValueError("Unknown scene or grid < 12 or spectral bins < 32")
    # Restrict acquisition to one Brillouin branch; no folding or interpolation.
    inst = Instrument(n_channels=bins, start_frequency_ghz=3.0,
                      end_frequency_ghz=8.5, photon_rate=2400,
                      read_noise_counts=0.0, dark_rate=0.12,
                      elastic_peak_counts=0.0, include_adjacent_orders=False,
                      spectrometer_gaussian_linewidth_ghz=0.18)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    radius = np.sqrt((x/0.86)**2 + (y/0.78)**2)
    inside = radius < 1
    nucleus = ((x-0.17)/0.32)**2 + ((y+0.1)/0.35)**2 < 1
    inclusion = ((x+0.34)**2 + (y-0.23)**2) < 0.12**2
    fraction = gaussian_filter(inside.astype(float), 0.7) * (0.82-0.12*x)
    domain = nucleus.astype(float)
    if name == "hydrogel_gradient":
        domain = 0.35 * np.sin(2.5*x) + 0.25 * np.cos(2*y)
        feature = inside & (x > 0.25)
        control = inside & (x < -0.25)
    else:
        domain = gaussian_filter(domain, 0.45)
        feature = binary_erosion(nucleus)
        control = inside & ~binary_dilation(nucleus, iterations=2)
    if name == "fine_structure":
        stripe = inside & (abs(x + 0.25 + 0.12*np.sin(5*y)) < 0.055)
        domain = domain + 0.8 * stripe + 1.2 * inclusion
        feature = stripe | inclusion
        control = binary_dilation(feature, iterations=2) & ~feature & inside
    centre = 5.61 + 0.075*x + 0.11*domain
    width = 0.37 + 0.055*(y+1) + 0.045*domain
    # Sub-resolution distributions and shoulders break the fixed-endmember
    # hypothesis. The underlying local curves are not single Lorentzians.
    shoulder = 0.025 + 0.11 * (domain-domain.min()) / max(np.ptp(domain), 1e-8)
    brightness = 0.85 + 0.25*(x+1)/2 + 0.08*np.cos(3*y)
    medium = Material("medium", bfs_ghz=5.49, linewidth_ghz=0.36)
    water = compose_densities(np.ones((1, 1)), [medium], inst, 1.0)[0]
    pure = np.empty((size, size, bins))
    # Reuse the existing simulator with pixel-specific line centres and widths.
    for iy in range(size):
        for ix in range(size):
            m = Material("local material", bfs_ghz=float(centre[iy, ix]),
                         linewidth_ghz=float(width[iy, ix]))
            s = replace(m, name="shoulder", bfs_ghz=m.bfs_ghz+0.27,
                        linewidth_ghz=m.linewidth_ghz*0.8)
            weights = np.array([[1-shoulder[iy, ix], shoulder[iy, ix]]])
            pure[iy, ix] = compose_densities(weights, [m, s], inst, 1.0)[0]
    rate = brightness[..., None] * (fraction[..., None]*pure
                                    + (1-fraction[..., None])*water)
    # Spatial optical PSF acts on the expected signal before photon sampling.
    rate = gaussian_filter(rate, sigma=(0.45, 0.45, 0)) + inst.dark_rate
    boundary = binary_dilation(inside) ^ binary_erosion(inside)
    boundary |= binary_dilation(nucleus) ^ binary_erosion(nucleus)
    return dict(rate=rate, frequency=inst.frequency_axis(), inside=inside,
                boundary=boundary, feature=feature, control=control,
                pure_material_rate=pure, medium_rate=water,
                material_fraction=fraction, instrument=inst.to_dict())


def sample_counts(rate, exposure, seed):
    if exposure <= 0:
        raise ValueError("Exposure must be positive")
    # rate already includes the dark current; do not add it a second time.
    detector = Instrument(dark_rate=0, read_noise_counts=0,
                          excess_noise_factor=1, baseline_offset=0)
    return apply_detector(np.asarray(rate)*exposure, exposure, detector,
                          np.random.default_rng(seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstruction_data"))
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--exposures", type=float, nargs="+", default=[16, 4, 1, .25, .0625])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        scene = make_scene(name, args.size, args.bins)
        np.savez_compressed(args.output / f"{name}_truth.npz",
                            **{k:v for k,v in scene.items() if k != "instrument"})
        for j, exposure in enumerate(args.exposures):
            np.savez_compressed(args.output / f"{name}_{exposure:g}s.npz",
                                counts=sample_counts(scene["rate"], exposure, 100+j),
                                frequency=scene["frequency"], exposure=exposure)


if __name__ == "__main__":
    main()
