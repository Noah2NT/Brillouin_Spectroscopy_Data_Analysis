# Current revision notes

## Changes requested by the author

- Preserved Section 1.1 and the useful cuts from the earlier revision.
- Rewrote the remaining introduction and methodology in plainer language, removing repeated explanations and shortening the study aims.
- Restored the earlier two-column List of Symbols, grouping related symbols rather than giving every symbol its own row with units and an equation reference. The grouped list fits on one page.
- Replaced the case codes throughout the paper, equations, tables and Figure 9 with descriptive names: fixed or varying sample, with known or unknown spectra.
- Kept the improved figure captions. Only Figure 9's caption needed adapting to the descriptive names.
- Retained the baseline, comparison and settings tables, with simpler headings and less repeated information.

## Scientific corrections retained

The intensity law still uses a fixed reference concentration of 0.10, so changing the tested range does not change the spectrum of a given solution. The focus position, common photon-count scale, bin integrals, distinction between collection and signal fractions, and defined recovery errors are retained. The regularisation convention and proposed numerical settings remain specified. The only equation change in this style pass is renaming the Wasserstein superscripts by their concentration/BFS coordinates; the mathematics is unchanged.

The coefficients remain prescribed synthetic inputs. The baseline water linewidth and BFS contrast form a favourable case; additional synthetic conditions test stronger overlap. The cited papers do not establish a matched 660 nm glycerol calibration. Proposed extra comparisons remain planned work, including Poisson-likelihood fitting and instrument, focus and lateral-averaging tests. Endpoint spectra were already mentioned in the earlier unknown-spectrum method; the revised wording makes the supplied ideal references explicit.

## Files and checks

The existing `main.tex` was edited in place. Figures 9, 12 and 13 now use the original Arial versions supplied by the author, with only the requested corrections. Their original page dimensions, axes rectangles and font sizes are preserved. The other 12 figure files were retained. All 15 figures remain, with no imposed page limit. The bibliography remains unchanged at 17 cited works.

The introduction and methodology now explain that the concentration fields are prescribed numerical tests, rather than a stable liquid specimen or a tissue-structure model. Each local solution produces one positive-shift peak; spatially separate contributions are combined by the collection model. The numerical coefficients remain explicitly illustrative. Although the reference paper's text could be read, downloading its graph for a verified numerical fit was blocked by the site's device check. No fitted coefficients were invented; that part still requires the source PDF or numerical concentration data.

Checks passed for references, removal of case codes, preservation of displayed equations and preservation of the other 14 figure captions. Diagnostic TeX/BibTeX compilation passed with no overfull boxes. The built-in editor compiler could not launch because Windows denied access to its executable, so the final PDF layout remains unverified. Existing `Thesis.pdf` and older ZIP files are previous exports; use the current source, bibliography and figure folder.
