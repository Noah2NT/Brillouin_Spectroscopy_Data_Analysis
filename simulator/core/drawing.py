"""The painted sample: a high-resolution map of which material is where.

The canvas is deliberately finer than the scan grid.  Setting a scan resolution
downsamples it, and each scan pixel picks up whatever mixture of materials its
footprint covers -- which is where sub-pixel material mixing comes from, and
why changing the scan resolution never requires redrawing.

Drawings are stored as compressed ``.npz`` because the index map is a large
array of small integers.  A 2000x2000 canvas is 4 MB raw and compresses to a
few tens of kilobytes, whereas the same data base64-encoded inside JSON would
be several megabytes of text.
"""

from __future__ import annotations

import json
import os

import numpy as np

#: File extension for saved drawings.
DRAWING_EXT = ".bcell.npz"

#: Canvas sub-cells per scan pixel when a drawing is created from a scan size.
DEFAULT_SUPERSAMPLE = 8


class CellDrawing:
    """A painted sample.

    Attributes
    ----------
    index_map : ndarray of uint8, shape (height, width)
        Material index for every canvas sub-cell.  0 is the background
        material.
    width_um, height_um : float
        Physical size of the whole canvas in microns.  This, not the pixel
        count, is what ties the drawing to a scan in real units.
    material_names : list of str
        Names of the materials this drawing was painted with, in index order.
        Saved so a drawing can be remapped if the library later changes.
    """

    def __init__(self, index_map, width_um, height_um, material_names=None):
        self.index_map = np.ascontiguousarray(index_map, dtype=np.uint8)
        self.width_um = float(width_um)
        self.height_um = float(height_um)
        self.material_names = list(material_names or [])

    # -- construction --------------------------------------------------------

    @classmethod
    def blank(cls, width_px, height_px, width_um, height_um, material_names=None):
        """An empty canvas filled with the background material."""
        return cls(np.zeros((int(height_px), int(width_px)), dtype=np.uint8),
                   width_um, height_um, material_names)

    @classmethod
    def for_scan(cls, grid_w, grid_h, step_um, supersample=DEFAULT_SUPERSAMPLE,
                 material_names=None):
        """A canvas sized to a scan grid at the given supersampling factor."""
        return cls.blank(int(grid_w) * int(supersample),
                         int(grid_h) * int(supersample),
                         grid_w * step_um, grid_h * step_um,
                         material_names)

    # -- geometry ------------------------------------------------------------

    @property
    def shape(self):
        return self.index_map.shape

    @property
    def width_px(self):
        return int(self.index_map.shape[1])

    @property
    def height_px(self):
        return int(self.index_map.shape[0])

    @property
    def um_per_subcell_x(self):
        return self.width_um / max(self.width_px, 1)

    @property
    def um_per_subcell_y(self):
        return self.height_um / max(self.height_px, 1)

    def material_counts(self, n_materials):
        """How many sub-cells belong to each material index."""
        return np.bincount(self.index_map.ravel(), minlength=int(n_materials))

    def used_indices(self):
        """Sorted material indices that actually appear on the canvas."""
        return np.unique(self.index_map).tolist()

    # -- editing -------------------------------------------------------------

    def copy(self):
        return CellDrawing(self.index_map.copy(), self.width_um, self.height_um,
                           list(self.material_names))

    def fill_all(self, index):
        self.index_map[:] = np.uint8(index)

    def remap(self, mapping):
        """Renumber material indices in place.

        Parameters
        ----------
        mapping : dict
            ``{old_index: new_index}``.  Indices not mentioned are left alone.
        """
        if not mapping:
            return
        lut = np.arange(256, dtype=np.uint8)
        for old, new in mapping.items():
            lut[int(old)] = np.uint8(new)
        self.index_map = lut[self.index_map]

    def resize_canvas(self, width_px, height_px):
        """Resample the index map to a new pixel size, preserving physical size.

        Nearest-neighbour, because interpolating material indices would invent
        materials that were never painted.
        """
        h, w = self.index_map.shape
        ys = np.clip((np.arange(int(height_px)) + 0.5) * h / int(height_px), 0, h - 1).astype(int)
        xs = np.clip((np.arange(int(width_px)) + 0.5) * w / int(width_px), 0, w - 1).astype(int)
        self.index_map = np.ascontiguousarray(self.index_map[np.ix_(ys, xs)])

    # -- persistence ---------------------------------------------------------

    def metadata(self):
        return {
            "version": 1,
            "width_um": self.width_um,
            "height_um": self.height_um,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "material_names": self.material_names,
        }

    def save(self, path):
        """Write to a compressed ``.npz``."""
        path = _with_extension(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(path,
                            index_map=self.index_map,
                            meta=json.dumps(self.metadata()))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            index_map = data["index_map"]
            meta = json.loads(str(data["meta"]))
        return cls(index_map, meta.get("width_um", index_map.shape[1]),
                   meta.get("height_um", index_map.shape[0]),
                   meta.get("material_names", []))

    def reconcile_with_library(self, library):
        """Remap indices so they line up with a library whose order has changed.

        Materials are matched by the names stored in the drawing.  Anything
        that no longer exists collapses to the background material, and the
        caller is told which names were lost.

        Returns
        -------
        list of str
            Names that could not be matched.
        """
        if not self.material_names:
            self.material_names = list(library.names)
            return []

        mapping = {}
        missing = []
        for old_index, name in enumerate(self.material_names):
            new_index = library.index_of(name)
            if new_index is None:
                missing.append(name)
                mapping[old_index] = 0
            elif new_index != old_index:
                mapping[old_index] = new_index
        self.remap(mapping)
        self.material_names = list(library.names)
        return missing


def _with_extension(path):
    if path.endswith(DRAWING_EXT):
        return path
    if path.endswith(".npz"):
        return path
    return path + DRAWING_EXT
