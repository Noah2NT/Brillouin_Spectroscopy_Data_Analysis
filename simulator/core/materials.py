"""Material definitions and the persistent material library.

A material is whatever a painted region is made of: culture medium, cytoplasm,
nucleus, a hydrogel, a bead.  What the simulator needs from it is a Brillouin
line (shift and width) and a couple of brightness weights.

Defaults are seeded from the vendor fit columns of the real scans in
``data/csv``, where the Brillouin shift runs 5.49-5.71 GHz and the linewidth
(FWHM) runs 0.51-0.78 GHz.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, replace

from . import physics

#: Where the user's material library lives by default.
DEFAULT_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "materials.json",
)

#: Shipped defaults, used when no library file exists yet.
BUILTIN_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "default_materials.json",
)


@dataclass
class Material:
    """One paintable substance.

    Attributes
    ----------
    name : str
        Unique display name; also the key used in saved drawings.
    color : str
        ``#rrggbb`` swatch used on the canvas.  Purely cosmetic.
    bfs_ghz : float
        Brillouin frequency shift.  Stored directly rather than derived,
        because that is how the values in this project are actually known.
    linewidth_ghz : float
        Intrinsic Brillouin linewidth, as a full width at half maximum.
        Note the instrument CSV stores a *half* width in its ``Half Width``
        column; the writer halves this on the way out.
    brillouin_efficiency : float
        Relative Brillouin scattering strength per unit volume.  A pixel's
        photon budget is shared between materials in proportion to
        ``area_fraction * brillouin_efficiency``.
    elastic_efficiency : float
        Relative elastic (Rayleigh) scattering strength.  Drives the height of
        the central line, which in real data is 20-60k counts.
    refractive_index, acoustic_velocity_ms, acoustic_attenuation_1pm : float or None
        Optional physical properties.  When ``refractive_index`` and
        ``acoustic_velocity_ms`` are both set, :meth:`derived_bfs` can compute
        the shift from thesis Eq. (1.1) instead of trusting ``bfs_ghz``.
    temperature_scaling : bool
        When true this material's Brillouin area scales with the scan
        temperature, following ``sigma = k_B * T * alpha_ac`` (thesis Eq. 3.12).
    notes : str
        Free text; round-trips through the library file.
    """

    name: str
    color: str = "#8899aa"
    bfs_ghz: float = 5.55
    linewidth_ghz: float = 0.62
    brillouin_efficiency: float = 1.0
    elastic_efficiency: float = 1.0
    refractive_index: float | None = None
    acoustic_velocity_ms: float | None = None
    acoustic_attenuation_1pm: float | None = None
    temperature_scaling: bool = True
    notes: str = ""

    def derived_bfs(self, wavelength_nm, scattering_angle_deg=180.0):
        """Shift computed from the optical properties, or ``None`` if unset."""
        if self.refractive_index is None or self.acoustic_velocity_ms is None:
            return None
        return physics.bfs_from_optics(
            self.refractive_index, self.acoustic_velocity_ms,
            wavelength_nm, scattering_angle_deg,
        )

    @property
    def hwhm_ghz(self):
        """Half width at half maximum, which the Lorentzian maths wants."""
        return self.linewidth_ghz / 2.0

    def derived_linewidth(self):
        """Intrinsic linewidth (FWHM) implied by the acoustic attenuation."""
        if self.acoustic_attenuation_1pm is None or self.acoustic_velocity_ms is None:
            return None
        return physics.acoustic_linewidth(
            self.acoustic_velocity_ms, self.acoustic_attenuation_1pm)

    def validate(self):
        """Return a list of human-readable problems; empty means the material is fine."""
        problems = []
        if not self.name.strip():
            problems.append("Name cannot be blank.")
        if not (0.0 < self.bfs_ghz < 100.0):
            problems.append(f"Brillouin shift {self.bfs_ghz} GHz is outside 0-100 GHz.")
        if not (0.0 < self.linewidth_ghz < 20.0):
            problems.append(f"Linewidth {self.linewidth_ghz} GHz is outside 0-20 GHz.")
        if self.brillouin_efficiency < 0:
            problems.append("Brillouin efficiency cannot be negative.")
        if self.elastic_efficiency < 0:
            problems.append("Elastic efficiency cannot be negative.")
        if not (self.color.startswith("#") and len(self.color) == 7):
            problems.append(f"Colour {self.color!r} is not in #rrggbb form.")
        return problems

    def copy_with(self, **changes):
        """Return a modified copy, leaving this instance alone."""
        return replace(self, **changes)


class MaterialLibrary:
    """An ordered, name-keyed collection of materials that persists to JSON.

    Index 0 is the background material -- whatever unpainted canvas is made of,
    normally the culture medium.  Drawings store material *indices*, so
    reordering the library reinterprets existing drawings; :meth:`move` is
    provided but the UI warns about it.
    """

    def __init__(self, materials=None):
        self._materials: list[Material] = list(materials or [])

    # -- container behaviour -------------------------------------------------

    def __len__(self):
        return len(self._materials)

    def __iter__(self):
        return iter(self._materials)

    def __getitem__(self, key):
        if isinstance(key, str):
            for m in self._materials:
                if m.name == key:
                    return m
            raise KeyError(key)
        return self._materials[key]

    def __contains__(self, name):
        return any(m.name == name for m in self._materials)

    @property
    def names(self):
        return [m.name for m in self._materials]

    def index_of(self, name):
        """Index of a material by name, or ``None`` if absent."""
        for i, m in enumerate(self._materials):
            if m.name == name:
                return i
        return None

    # -- mutation ------------------------------------------------------------

    def add(self, material):
        """Append a material, renaming it if the name collides."""
        material = material.copy_with(name=self._unique_name(material.name))
        self._materials.append(material)
        return material

    def update(self, index, material):
        """Replace the material at ``index``, keeping names unique."""
        others = [m.name for i, m in enumerate(self._materials) if i != index]
        if material.name in others:
            material = material.copy_with(name=self._unique_name(material.name, others))
        self._materials[index] = material
        return material

    def remove(self, index):
        """Delete a material.  The background material (index 0) is protected."""
        if index == 0:
            raise ValueError("The background material cannot be removed.")
        return self._materials.pop(index)

    def move(self, index, new_index):
        """Reorder the library.  Existing drawings refer to indices, so callers
        are responsible for remapping any affected drawing."""
        m = self._materials.pop(index)
        self._materials.insert(new_index, m)

    def _unique_name(self, name, taken=None):
        taken = set(taken if taken is not None else self.names)
        if name not in taken:
            return name
        n = 2
        while f"{name} {n}" in taken:
            n += 1
        return f"{name} {n}"

    # -- persistence ---------------------------------------------------------

    def to_dict(self):
        return {"version": 1, "materials": [asdict(m) for m in self._materials]}

    @classmethod
    def from_dict(cls, data):
        fields = {f for f in Material.__dataclass_fields__}
        mats = []
        for raw in data.get("materials", []):
            mats.append(Material(**{k: v for k, v in raw.items() if k in fields}))
        return cls(mats)

    def save(self, path=DEFAULT_LIBRARY_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path=DEFAULT_LIBRARY_PATH, merge_builtins=True):
        """Load the user's library, falling back to the shipped defaults.

        Once the app has been run, the user's own library file shadows the
        shipped one.  That is right for materials they have edited, but it
        would also mean a corrected or newly added default never reaches them.
        So any shipped material whose *name* is absent from the user's library
        is appended on load.  Existing entries are never overwritten -- use
        :meth:`restore_builtin` for that.
        """
        library = None
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                library = cls.from_dict(json.load(fh))
        if library is None:
            return cls.builtin()
        if merge_builtins:
            library.merge_missing(cls.builtin())
        return library

    @classmethod
    def builtin(cls):
        """The shipped library, from disk if present, otherwise from code."""
        if os.path.exists(BUILTIN_LIBRARY_PATH):
            with open(BUILTIN_LIBRARY_PATH, encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        return cls(default_materials())

    def merge_missing(self, other):
        """Append materials from ``other`` whose names are not already here.

        Returns the names that were added.
        """
        added = []
        for material in other:
            if material.name not in self:
                self._materials.append(material)
                added.append(material.name)
        return added

    def restore_builtin(self):
        """Replace every shipped material with its original definition.

        User-created materials -- names not in the shipped set -- are kept.
        Returns the names that were reset.
        """
        reset = []
        for material in self.builtin():
            index = self.index_of(material.name)
            if index is None:
                self._materials.append(material)
            else:
                self._materials[index] = material
            reset.append(material.name)
        return reset


def default_materials():
    """The shipped material set.

    Brillouin shifts and widths are taken from the vendor fit columns of the
    real scans in ``data/csv`` so that synthetic files land in the same range
    as measured ones.  Optical properties are literature values and are only
    used by the optional Eq. (1.1) calculator.
    """
    return [
        Material(
            name="Culture medium", color="#1f3a5f",
            bfs_ghz=5.510, linewidth_ghz=0.254,
            brillouin_efficiency=1.0, elastic_efficiency=1.0,
            refractive_index=1.334, acoustic_velocity_ms=1487.0,
            notes="Background. Matches the water-like peak in water_plexiglass.csv.",
        ),
        Material(
            name="Cytoplasm", color="#4a90c2",
            bfs_ghz=5.560, linewidth_ghz=0.340,
            brillouin_efficiency=1.15, elastic_efficiency=1.1,
            refractive_index=1.360, acoustic_velocity_ms=1472.0,
            notes="Typical of the bulk cell body across the data/csv scans.",
        ),
        Material(
            name="Nucleus", color="#e0a030",
            bfs_ghz=5.650, linewidth_ghz=0.394,
            brillouin_efficiency=1.35, elastic_efficiency=1.2,
            refractive_index=1.375, acoustic_velocity_ms=1479.0,
            notes="Stiffer and brighter than cytoplasm; the usual segmentation target.",
        ),
        Material(
            name="Nucleolus", color="#c9542a",
            bfs_ghz=5.720, linewidth_ghz=0.434,
            brillouin_efficiency=1.5, elastic_efficiency=1.25,
            refractive_index=1.390, acoustic_velocity_ms=1481.0,
            notes="Small high-shift inclusion, useful for testing spatial resolution.",
        ),
        Material(
            name="Lipid droplet", color="#7cb342",
            bfs_ghz=5.380, linewidth_ghz=0.298,
            brillouin_efficiency=0.9, elastic_efficiency=1.4,
            refractive_index=1.460, acoustic_velocity_ms=1327.0,
            notes="Low-shift inclusion; tests whether an algorithm can resolve both signs.",
        ),
        Material(
            name="Collagen fibre", color="#9c6ade",
            bfs_ghz=5.950, linewidth_ghz=0.486,
            brillouin_efficiency=1.2, elastic_efficiency=1.3,
            refractive_index=1.430, acoustic_velocity_ms=1498.0,
            notes="Stiff extracellular structure; well separated from the cell body.",
        ),
        Material(
            name="Polystyrene dish", color="#ef9a9a",
            bfs_ghz=10.380, linewidth_ghz=0.300,
            brillouin_efficiency=1.0, elastic_efficiency=1.5,
            refractive_index=1.590, acoustic_velocity_ms=2351.0,
            temperature_scaling=False,
            notes="Culture plate under the cells. Most Plate2/Plate3 scans in "
                  "data/csv show a weak, near-uniform peak at 10.4-10.6 GHz, "
                  "which is where polystyrene lands at this wavelength.",
        ),
        Material(
            name="Plexiglass (PMMA)", color="#b0bec5",
            bfs_ghz=11.750, linewidth_ghz=0.188,
            brillouin_efficiency=2.5, elastic_efficiency=2.0,
            refractive_index=1.492, acoustic_velocity_ms=2835.0,
            notes="Shift measured directly from water_plexiglass.csv, where the peak "
                  "at 11.75 GHz is anti-correlated with the water peak at r = -0.96.",
        ),
        Material(
            name="Glass substrate", color="#546e7a",
            bfs_ghz=24.860, linewidth_ghz=0.158,
            brillouin_efficiency=1.8, elastic_efficiency=3.0,
            refractive_index=1.517, acoustic_velocity_ms=5900.0,
            temperature_scaling=False,
            notes="Coverslip. Its Brillouin line sits far outside the +-14.9 GHz "
                  "window, so it contributes elastic signal only - which is what a "
                  "real coverslip does.",
        ),
    ]


def write_builtin_library(path=BUILTIN_LIBRARY_PATH):
    """Write the shipped defaults to disk.  Called once at package setup."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lib = MaterialLibrary(default_materials())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(lib.to_dict(), fh, indent=2)
    return path
