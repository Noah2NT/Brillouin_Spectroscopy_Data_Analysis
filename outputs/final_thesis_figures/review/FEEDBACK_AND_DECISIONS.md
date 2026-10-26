# All thesis figures for consolidated review

The review numbers below are stable discussion IDs, not final LaTeX numbering. There are twelve original figure slots and two additional simulation plots. Some slots could be combined once Noah has reviewed them. No new designs will be generated until the consolidated suggestions arrive.

Style A is locked in: Arial, white backgrounds, clean vector artwork, blue/orange comparisons and neutral equipment. Figure 1 is approved. Figure 2 v2 is a draft. For Figures 3–12, the boards show the original placeholders. Figures 13–14 show the forward-model plots produced with the revised thesis.

Supervisor comments are labelled separately from proposed design changes. Where a supervisor commented on surrounding prose rather than the figure itself, that distinction is retained. The full source feedback ledger is `../../thesis_revision/Supervisor_feedback_summary.md`.

| Review ID | Figure | Relevant supervisor feedback | Proposed treatment to discuss |
|---|---|---|---|
| 1 | Cell mechanics | IK1: the duplicated cell images are unclear/repetitive; use one and clarify the intended comparison. | Already approved: one cell/probe diagram with two force–indentation curves. Retain unless Noah asks for another change. |
| 2 | Brillouin acquisition and scan | IK1: correct acoustic-wave orientation; identify shifted light by frequency or colour; separate detection from the objective; include theta. Chris: refer to the figure in the image-formation text. | v2 includes axial wave motifs and expanding dashed guides from the selected pixel to its spectrum. Noah disliked flat bands, disconnected arrows and the weak b-to-c link in v1. Further comments pending. |
| 3 | Brillouin spectral features | Chris's nearby notes request clearer explanation of Stokes/anti-Stokes and merged linewidth/damping discussion. IK1's nearby notes emphasise instrumental broadening and characterising the instrument response before correction. These are principally explanatory-text comments. | Redraw labelled Rayleigh/Stokes/anti-Stokes spectrum, BFS and FWHM, plus small examples distinguishing a centre shift from broadening. Do not imply that every measured linewidth is intrinsic damping. |
| 4 | Water and Plexiglass | Chris: explicitly refer to the figure when explaining separated and overlapping material responses. | Schematic separated peaks with clear material labels and frequency/intensity axes. These are introductory examples, not the water–glycerol simulation dataset. |
| 5 | Collection volume and mixing | Chris: refer explicitly to the collection-volume illustration. | Show the finite axial collection region passing through sample and medium, their local spectra, and their weighted sum. Simplify the sketches into one clear flow. |
| 6 | Noise | Chris: refer to Figure 6, merge the discussion and explain different noise sources; in methodology he asks whether only shot noise will be studied. | Keep the same underlying spectrum at several declared shot-noise levels, with common axes and a clearly labelled noiseless reference. The present protocol has not been expanded to other noise sources. |
| 7 | Single-peak fitting | Chris: use the figure to illustrate that a good single-peak fit need not recover the underlying components; avoid repeated Lorentzian introductions. | Distinguish the two true components, their sum and a single fitted peak. Make the fitted centre and resulting apparent linewidth easy to compare. |
| 8 | Known/unknown unmixing | Chris: rephrase and merge the known-spectrum explanation; introduce PCA and related methods in a new subsection. These comments concern the surrounding text. | Retain the known-versus-unknown workflow comparison, but replace the placeholder's optical-wavelength axes with Brillouin frequency axes. The axis correction is a design/technical recommendation from this review, not a quoted supervisor annotation. |
| 9 | Four configurations | Chris: make the research aims more specific and remove repetition; his mark beside the aims list is positive. No separate graphical correction was found for this diagram. | A consistent 2-by-2 matrix: fixed/varying responses versus known/unknown spectra, with the recovery target stated in each cell. |
| 10 | Fixed/varying sample models | Chris: merge the descriptions and reference the sample figure. | Two matching cross-sections using flat fill for the uniform solution and a concentration colour map for the prescribed gradient. Clearly distinguish model concentration from optical signal. Could be retained as a concise sample-definition figure even if Figure 14 also shows the geometry. |
| 11 | Fixed mixture formation | No separate figure-specific annotation was found. Relevant general advice is to connect diagrams explicitly to the equations and text. | Show one collection ray crossing water and a homogeneous glycerol solution, two local responses, their weighting and the summed spectrum. The 10% solution is one homogeneous spectral component. |
| 12 | Varying concentration formation | No separate figure-specific annotation was found. Chris's adjacent methodology comments ask about noise scope and the convolution symbol, not a correction to this drawing. | Show changing local spectra along the collection axis and their integrated response. Distinguish variation with depth in this figure from lateral scan-position variation in Figure 14. |
| 13 | Concentration curves | Noah requested BFS, linewidth and intensity versus concentration. In-person advice: state 0–10% is an initial range and replace concentration symbol c. | Restyle the three panels in A; keep the provisional-model status explicit. Solid intensity curve is constant area; dashed +/-20% curves are sensitivity tests, not measurements. |
| 14 | Noiseless dataset | Noah requested a view of what the simulation data will look like and confirmed it should be noiseless. No supervisor annotation exists for this newly generated plot. | Current plot shows fixed/varying sample truth, a central spectral line scan and selected spectra. Ask whether to keep all six panels or simplify. Retain common scales and distinguish prescribed concentration maps from measured/reconstructed maps. |

## Feedback template

Use the review ID followed by any requested changes. `Keep` is enough for a figure that already communicates the intended point. Comments can cover content, layout, labels, relative sizes, colours, or whether figures should be combined.

1. Approved / any additional change:
2. Further changes to v2:
3. Spectral features:
4. Water/Plexiglass:
5. Collection volume:
6. Noise:
7. Single-peak fit:
8. Unmixing:
9. Four configurations:
10. Sample models:
11. Fixed mixture:
12. Varying mixture:
13. Concentration curves:
14. Noiseless dataset:

Supervisor sources: `Honors_Thesis_Draft_IK1.pdf`, `Honors_Thesis_Draft_comments_cgp.pdf`, and Noah's account of the in-person discussion. Interpretations and new design proposals above are not presented as quotations or additional supervisor instructions.
