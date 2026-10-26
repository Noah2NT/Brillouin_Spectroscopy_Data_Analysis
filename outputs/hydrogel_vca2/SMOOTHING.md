# Smoothed hydrogel visualization

Uses exactly the VCA background <97% mask (301 pixels), but heights are the
background-plus-individual-Lorentzian hydrogel shares, not VCA abundances.
Gaussian smoothing with sigma 1.2 pixels acts on the height grid with zero outside
the mask. It rounds local height changes and tapers edges; it does not fit a sphere
or conserve total height/volume. Original spectral results are unchanged.

Each retained pixel retains its original centre and exact 0.92 x 0.92 column
footprint (area 0.8464 pixel-index units squared). No mask expansion or pixel
resizing. The original 0.08 gaps therefore remain: this is a rounded tiled shape,
not a watertight continuous sphere. Each tile is subdivided into 4 x 4 patches;
heights are interpolated bilinearly from the smoothed grid. Upper and lower
surfaces are mirrored about Z=0. Z is spectral percentage, not physical distance.

Run: `python analysis/pca_vca/hydrogel_smooth_ball.py`
