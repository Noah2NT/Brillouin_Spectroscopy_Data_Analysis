"""The painting surface.

Behaves like an indexed-colour paint program: every canvas cell holds a
material index, and the palette maps indices to colours.  The canvas is finer
than the scan grid, so a scan pixel covering several cells picks up a mixture.

Implementation notes worth knowing before editing this file:

* Shapes are rasterised by letting ``QPainter`` draw white onto a throwaway
  ``Format_Grayscale8`` image and then using that as a numpy mask.  Painting
  directly into a ``Format_Indexed8`` image does *not* work -- Qt treats the
  pen colour as an RGB value rather than a palette index.
* Antialiasing is deliberately off.  Material indices are categorical, so a
  half-blended index would mean a material nobody chose.  Sub-pixel mixing
  comes from downsampling the fine canvas, which is the physically meaningful
  route.
* The display image is cached and only rebuilt when the index map changes,
  because rebuilding an RGB image of a large canvas on every repaint is
  noticeably slow.
"""

from __future__ import annotations

import zlib

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage

#: Tool identifiers.
PENCIL, LINE, RECT, ELLIPSE, POLYGON, FILL, ERASER, PICKER = (
    "pencil", "line", "rect", "ellipse", "polygon", "fill", "eraser", "picker")

#: How many undo states to keep.  States are zlib-compressed, and index maps
#: compress extremely well, so this is cheap even for a large canvas.
UNDO_DEPTH = 40


class CanvasWidget(QtWidgets.QWidget):
    """Indexed-colour painting surface with zoom, pan and undo."""

    drawing_changed = QtCore.Signal()
    material_picked = QtCore.Signal(int)
    cursor_moved = QtCore.Signal(float, float, int)  # x_um, y_um, material index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setMinimumSize(320, 320)

        self._drawing = None
        self._colors = np.zeros((256, 3), dtype=np.uint8)
        self._cache = None            # cached QImage of the whole canvas
        self._cache_dirty = True

        self.tool = PENCIL
        self.material_index = 1
        self.brush_size = 6
        self.filled_shapes = True

        self.zoom = 1.0
        self.offset = QtCore.QPointF(0.0, 0.0)
        self.show_scan_grid = True
        self.scan_grid = (40, 40)

        self._undo, self._redo = [], []
        self._dragging = False
        self._panning = False
        self._last_pan = None
        self._stroke_start = None
        self._preview_end = None
        self._polygon = []

    # -- model ---------------------------------------------------------------

    def set_drawing(self, drawing):
        self._drawing = drawing
        self._undo.clear()
        self._redo.clear()
        self._cache_dirty = True
        self.fit_to_view()
        self.update()

    @property
    def drawing(self):
        return self._drawing

    def set_palette(self, materials):
        """Update the index-to-colour table from the material library."""
        self._colors[:] = 40
        for i, m in enumerate(materials):
            if i > 255:
                break
            c = QtGui.QColor(m.color)
            self._colors[i] = (c.red(), c.green(), c.blue())
        self._cache_dirty = True
        self.update()

    def set_scan_grid(self, grid_w, grid_h):
        self.scan_grid = (max(int(grid_w), 1), max(int(grid_h), 1))
        self.update()

    # -- undo ----------------------------------------------------------------

    def push_undo(self):
        if self._drawing is None:
            return
        blob = zlib.compress(self._drawing.index_map.tobytes(), 1)
        self._undo.append((blob, self._drawing.index_map.shape))
        if len(self._undo) > UNDO_DEPTH:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, blob_shape):
        blob, shape = blob_shape
        return np.frombuffer(zlib.decompress(blob), dtype=np.uint8).reshape(shape).copy()

    def undo(self):
        if not self._undo or self._drawing is None:
            return
        current = (zlib.compress(self._drawing.index_map.tobytes(), 1),
                   self._drawing.index_map.shape)
        self._redo.append(current)
        self._drawing.index_map = self._restore(self._undo.pop())
        self._after_edit()

    def redo(self):
        if not self._redo or self._drawing is None:
            return
        current = (zlib.compress(self._drawing.index_map.tobytes(), 1),
                   self._drawing.index_map.shape)
        self._undo.append(current)
        self._drawing.index_map = self._restore(self._redo.pop())
        self._after_edit()

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def _after_edit(self):
        self._cache_dirty = True
        self.update()
        self.drawing_changed.emit()

    # -- coordinate transforms ----------------------------------------------

    def _canvas_rect(self):
        if self._drawing is None:
            return QtCore.QRectF()
        w = self._drawing.width_px * self.zoom
        h = self._drawing.height_px * self.zoom
        return QtCore.QRectF(self.offset.x(), self.offset.y(), w, h)

    def widget_to_canvas(self, pos):
        """Widget coordinates to canvas cell coordinates (floats)."""
        if self._drawing is None or self.zoom <= 0:
            return (0.0, 0.0)
        return ((pos.x() - self.offset.x()) / self.zoom,
                (pos.y() - self.offset.y()) / self.zoom)

    def fit_to_view(self):
        if self._drawing is None:
            return
        margin = 16
        zx = (self.width() - margin) / max(self._drawing.width_px, 1)
        zy = (self.height() - margin) / max(self._drawing.height_px, 1)
        self.zoom = max(min(zx, zy), 0.02)
        self.offset = QtCore.QPointF(
            (self.width() - self._drawing.width_px * self.zoom) / 2.0,
            (self.height() - self._drawing.height_px * self.zoom) / 2.0)
        self.update()

    # -- painting operations -------------------------------------------------

    def _shape_mask(self, painter_ops):
        """Rasterise shapes to a boolean mask via a grayscale scratch image."""
        h, w = self._drawing.index_map.shape
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_Grayscale8)
        img.fill(0)
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter_ops(p)
        p.end()
        stride = img.bytesPerLine()
        buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=stride * h)
        return buf.reshape(h, stride)[:, :w] > 127

    def _pen(self, width):
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setWidth(max(int(width), 1))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        return pen

    def _apply(self, mask, index=None):
        idx = self.material_index if index is None else index
        self._drawing.index_map[mask] = np.uint8(idx)

    def stroke_segment(self, x0, y0, x1, y1, index=None):
        def ops(p):
            p.setPen(self._pen(self.brush_size))
            p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
        self._apply(self._shape_mask(ops), index)

    def draw_rect(self, x0, y0, x1, y1, index=None):
        rect = QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1)).normalized()

        def ops(p):
            p.setPen(self._pen(self.brush_size))
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255))
                       if self.filled_shapes else QtCore.Qt.NoBrush)
            p.drawRect(rect)
        self._apply(self._shape_mask(ops), index)

    def draw_ellipse(self, x0, y0, x1, y1, index=None):
        rect = QtCore.QRectF(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1)).normalized()

        def ops(p):
            p.setPen(self._pen(self.brush_size))
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255))
                       if self.filled_shapes else QtCore.Qt.NoBrush)
            p.drawEllipse(rect)
        self._apply(self._shape_mask(ops), index)

    def draw_polygon(self, points, index=None):
        poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])

        def ops(p):
            p.setPen(self._pen(self.brush_size))
            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255))
                       if self.filled_shapes else QtCore.Qt.NoBrush)
            p.drawPolygon(poly)
        self._apply(self._shape_mask(ops), index)

    def flood_fill(self, x, y, index=None):
        """Replace the connected same-material region containing (x, y)."""
        h, w = self._drawing.index_map.shape
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            return
        target = self._drawing.index_map[yi, xi]
        new = self.material_index if index is None else index
        if target == new:
            return
        labels, _ = ndimage.label(self._drawing.index_map == target)
        self._drawing.index_map[labels == labels[yi, xi]] = np.uint8(new)

    # -- events --------------------------------------------------------------

    def mousePressEvent(self, event):
        if self._drawing is None:
            return
        x, y = self.widget_to_canvas(event.position())

        if event.button() == QtCore.Qt.MiddleButton or (
                event.button() == QtCore.Qt.LeftButton
                and event.modifiers() & QtCore.Qt.ShiftModifier):
            self._panning = True
            self._last_pan = event.position()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return

        if event.button() == QtCore.Qt.RightButton:
            self._pick(x, y)
            return

        if self.tool == PICKER:
            self._pick(x, y)
            return

        if self.tool == POLYGON:
            self._polygon.append((x, y))
            self.update()
            return

        self.push_undo()
        self._dragging = True
        self._stroke_start = (x, y)
        self._preview_end = (x, y)

        if self.tool in (PENCIL, ERASER):
            idx = 0 if self.tool == ERASER else None
            self.stroke_segment(x, y, x, y, idx)
            self._after_edit()
        elif self.tool == FILL:
            self.flood_fill(x, y)
            self._dragging = False
            self._after_edit()

    def mouseMoveEvent(self, event):
        if self._drawing is None:
            return
        x, y = self.widget_to_canvas(event.position())
        self._emit_cursor(x, y)

        if self._panning and self._last_pan is not None:
            delta = event.position() - self._last_pan
            self.offset += delta
            self._last_pan = event.position()
            self.update()
            return

        if not self._dragging:
            return

        if self.tool in (PENCIL, ERASER):
            x0, y0 = self._stroke_start
            idx = 0 if self.tool == ERASER else None
            self.stroke_segment(x0, y0, x, y, idx)
            self._stroke_start = (x, y)
            self._after_edit()
        else:
            self._preview_end = (x, y)
            self.update()

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self._last_pan = None
            self.setCursor(QtCore.Qt.CrossCursor)
            return
        if self._drawing is None or not self._dragging:
            return
        x, y = self.widget_to_canvas(event.position())
        x0, y0 = self._stroke_start
        if self.tool == LINE:
            self.stroke_segment(x0, y0, x, y)
        elif self.tool == RECT:
            self.draw_rect(x0, y0, x, y)
        elif self.tool == ELLIPSE:
            self.draw_ellipse(x0, y0, x, y)
        self._dragging = False
        self._preview_end = None
        self._after_edit()

    def mouseDoubleClickEvent(self, event):
        if self.tool == POLYGON and len(self._polygon) >= 3:
            self.push_undo()
            self.draw_polygon(self._polygon)
            self._polygon = []
            self._after_edit()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self._polygon = []
            self._dragging = False
            self._preview_end = None
            self.update()
        elif event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.mouseDoubleClickEvent(event)
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        if self._drawing is None:
            return
        before = self.widget_to_canvas(event.position())
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        self.zoom = float(np.clip(self.zoom * factor, 0.02, 64.0))
        after = self.widget_to_canvas(event.position())
        self.offset += QtCore.QPointF((after[0] - before[0]) * self.zoom,
                                      (after[1] - before[1]) * self.zoom)
        self.update()

    def _pick(self, x, y):
        h, w = self._drawing.index_map.shape
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h:
            self.material_picked.emit(int(self._drawing.index_map[yi, xi]))

    def _emit_cursor(self, x, y):
        d = self._drawing
        h, w = d.index_map.shape
        xi, yi = int(x), int(y)
        idx = int(d.index_map[yi, xi]) if (0 <= xi < w and 0 <= yi < h) else -1
        self.cursor_moved.emit(x * d.um_per_subcell_x, y * d.um_per_subcell_y, idx)

    # -- rendering -----------------------------------------------------------

    def _rebuild_cache(self):
        rgb = self._colors[self._drawing.index_map]
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        self._cache = QtGui.QImage(rgb.data, w, h, 3 * w,
                                   QtGui.QImage.Format.Format_RGB888).copy()
        self._cache_dirty = False

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#141821"))
        if self._drawing is None:
            p.setPen(QtGui.QColor("#8892a4"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       "No drawing loaded.\nFile > New sample")
            return

        if self._cache_dirty or self._cache is None:
            self._rebuild_cache()

        target = self._canvas_rect()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self.zoom < 1.0)
        p.drawImage(target, self._cache)

        p.setPen(QtGui.QPen(QtGui.QColor("#3d4657"), 1))
        p.drawRect(target)

        if self.show_scan_grid:
            self._draw_scan_grid(p, target)
        self._draw_preview(p)

    def _draw_scan_grid(self, p, target):
        gw, gh = self.scan_grid
        if gw * gh > 40000:
            return
        step_x = target.width() / gw
        step_y = target.height() / gh
        if min(step_x, step_y) < 3:
            return
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 46), 1))
        for i in range(1, gw):
            x = target.left() + i * step_x
            p.drawLine(QtCore.QPointF(x, target.top()),
                       QtCore.QPointF(x, target.bottom()))
        for j in range(1, gh):
            y = target.top() + j * step_y
            p.drawLine(QtCore.QPointF(target.left(), y),
                       QtCore.QPointF(target.right(), y))

    def _draw_preview(self, p):
        """Rubber band for shape tools and the in-progress polygon."""
        p.setPen(QtGui.QPen(QtGui.QColor("#ffd166"), 1, QtCore.Qt.DashLine))
        p.setBrush(QtCore.Qt.NoBrush)

        def to_widget(pt):
            return QtCore.QPointF(self.offset.x() + pt[0] * self.zoom,
                                  self.offset.y() + pt[1] * self.zoom)

        if self._dragging and self._preview_end and self.tool in (LINE, RECT, ELLIPSE):
            a, b = to_widget(self._stroke_start), to_widget(self._preview_end)
            if self.tool == LINE:
                p.drawLine(a, b)
            elif self.tool == RECT:
                p.drawRect(QtCore.QRectF(a, b).normalized())
            else:
                p.drawEllipse(QtCore.QRectF(a, b).normalized())

        if self._polygon:
            pts = [to_widget(pt) for pt in self._polygon]
            p.drawPolyline(QtGui.QPolygonF(pts))
            for pt in pts:
                p.drawEllipse(pt, 3, 3)
