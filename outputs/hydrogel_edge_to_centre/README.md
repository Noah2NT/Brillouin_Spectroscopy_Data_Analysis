# Background-subtracted edge-to-centre spectra

301 pixels selected by VCA background <97%, ordered by Euclidean distance inside
the mask and grouped in six distance bands. Spatial distance is not concentration.

Corrected data = measured counts - fitted scaled background - fitted constant baseline.
The single displayed Lorentzian is the component from the saved joint fit, not an
independent refit: its amplitude, centre and width were fitted per pixel alongside
the background scale and constant baseline. No full mixture curves are displayed.
The corrected residual equals the original joint-fit residual (verified numerically).

Normalize corrected data and its Lorentzian by that Lorentzian's channel sum.
This preserves negative noise and avoids normalizing by a noisy residual sum.
Band means weight pixels equally. Peak and width summaries exclude nonconverged
fits; error bars show quartiles, not confidence intervals. No smoothing is applied.
The subtraction depends on the fit model, and cannot independently establish a
pure hydrogel spectrum or polymer concentration.

An external HTML header holds the legend, outside the plot title/axis layout.
Reproduce: python analysis/pca_vca/hydrogel_edge_to_centre.py
