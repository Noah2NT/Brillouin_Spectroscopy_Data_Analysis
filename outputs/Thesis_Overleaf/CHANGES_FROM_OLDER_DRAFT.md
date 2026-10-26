# Exact change record: older attached draft to current main.tex

Updated 18 October 2026. This compares the source attached with your collated-feedback request, before the formal rewrite you disliked, with the current saved main.tex. The baseline begins “The appearance of a cell does not always tell us how it behaves.” It is **not** the disliked intermediate version. On 18 October, Figure 9's original schematic distribution in panel C was restored at your request, retaining the corrected linewidth-versus-BFS panel D. Its caption now identifies the distribution as schematic; the methodology and its 51-node grid are unchanged.

The baseline is preserved in comparison/older_main.tex. Every source change, including punctuation, comments, captions and deleted wording, appears in comparison/main_changes.html and comparison/exact_changes.diff. Those files are the exhaustive record; this document explains the effects of the changes.

## What stayed the same

- Section 1.1's prose is identical to your attached source. Its Figure 1 caption and image width changed.
- Same title, author, student number, course, A4 paper, 11-point body text and 2 cm margins.
- Same four study cases, using descriptive names rather than C1–C4 codes; same NNLS, PCA, VCA and MCR-ALS methods.
- Same two-column, grouped List of Symbols style; definitions were updated and expanded.
- All 15 figure filenames and their order are retained. Their captions changed; several image contents changed as listed below.
- Same initial 0–10% glycerol-by-mass range, with revision after pilot work allowed before final evaluation.
- Same positive-shift Lorentzian model, concentration-dependent BFS and intrinsic-linewidth laws, Poisson shot noise, and main baseline values below.
- Same simulation scope: no experimental data collection, depth reconstruction or modulus estimation. Your older source already described mathematical sample fields and one peak per homogeneous local solution; this explanation is now more prominent.

## Layout, structure and language

1. Contents now precede the List of Symbols. The symbols list and References are added to the contents.
2. The grouped symbol table has 25 definition rows instead of 21, with the same two column widths. It adds the incident optical frequency, fixed reference concentration, focus, bin integrals, measured data, fraction conventions, dictionaries, evaluation sets and Figure 1 symbols. Related mechanical moduli are combined in one row.
3. Introduction subsection headings change from five to four. The old sections on mixed pixels and unmixing are combined under “Mixed spectra and spectral unmixing”, with three subheadings: mixing, known/unknown spectra, and continuously varying composition.
4. “What Brillouin microscopy measures” becomes “Brillouin spectra and their features”. The explanation is shorter; the peak features are introduced before their use.
5. “Research question and scope” becomes “Research question, aims and scope”, with four explicit aims. The initial concentration-range statement moves here.
6. The introduction and methods use shorter explanations and remove repeated warnings already covered elsewhere. “Plexiglass” becomes the chemical name poly(methyl methacrylate), PMMA.
7. The methodology still has the same seven main subsections. Three headings are shortened/updated: “Sample composition and component spectra”, “Photon statistics and comparison settings”, and “Implementation and reproducibility”. “Analysis of the four configurations” becomes “Analysis of the four cases”.
8. There are three numbered methodology tables instead of one: an expanded baseline table, a table of supplied information/recovery targets, and a table of extra tests and fitting settings. The baseline table gains a Symbol column. The symbol-list longtable no longer advances the numbered table count.
9. The appendix keeps the same scientific explanation and its two equations, with small wording/unit changes. Its equations are now A.1 and A.2, instead of continuing the main sequence.
10. Added the siunitx and doi packages, an upright differential command, consistent unit formatting and day–month–year dates. Bibliography style changes from apalike to plainnat to display DOI information. The source's separator-comment style remains.

## Baseline values: unchanged numerically

| Quantity | Both versions |
|---|---|
| Concentration | 0 to 0.10 mass fraction |
| Temperature; wavelength; angle | 296 K; 660 nm; 180° |
| Sphere and lateral scan | Radius 20 µm; 51 × 51 positions; 1 µm spacing; x and y from −25 to 25 µm |
| Axial collection | z = 0 to 30 µm; normalised uniform weighting |
| Depth integration | 0.25 µm midpoint spacing; 120 baseline points |
| Frequency centres | 401 centres, 4.50–8.50 GHz in 0.01 GHz steps |
| Water BFS; intrinsic linewidth; area | 6.05 GHz; 0.20 GHz; A₀ = 1 |
| BFS, linewidth and area coefficients | 0.85; 2.0; +0.20 |
| Instrument | Unit-area Lorentzian; FWHM 0.276 GHz |
| Baseline concentration exponent | β = 1 |
| Noise | Poisson only; zero additive background |
| Photon budgets | 10², 10³, 10⁴, 10⁵, plus a noiseless condition |
| Repetitions | 30 noise realisations per condition |
| Central fixed-ray collection fractions | 1/3 water and 2/3 solution |
| Central full-band signal fractions | Approximately 29.4% water and 70.6% solution |

The 10% endpoint remains BFS 6.56425 GHz, intrinsic linewidth 0.275445 GHz and area 1.20 A₀. The new prose rounds the first two to 6.564 and 0.275 GHz. That is a display change, not a changed calculation.

## Equation changes: what affects the model

There were **23 numbered main-text equations plus two appendix equations**. There are now **24 main-text equations plus two appendix equations**. The extra main-text number comes from writing the two Wasserstein coordinates separately. Several existing blocks contain more explicit definitions, so equation count alone does not describe the changes.

### 1. The intensity law changes away from the original 10% range

Old: A(wg) = A₀[1 + δA wg/wmax].

Current: A(wg) = A₀[1 + αA wg/wref], with wref = 0.10.

This is a real model correction. At the original maximum of 10% the laws agree. When the tested maximum changes, the spectrum of a given concentration now stays fixed. For example, 2.5% solution has area 1.05 A₀ even in a run whose maximum is 2.5%; the old formula would have given 1.20 A₀ in that run. The coefficient rename alone is cosmetic; the fixed denominator is substantive.

### 2. The known-spectrum distribution penalty changes

Old: raw squared residual + η times squared second differences of positive-bin masses.

Current: squared residual divided by the squared norm of the water reference column + η times a concentration-spacing-aware penalty on positive-bin density.

The new operator uses second differences of u/Δw, weighted by √Δw, giving matrix scaling (Δw)^−5/2. Water at zero concentration remains unpenalised, as before. Component spectra retain their relative areas.

This is a real change to the regularisation convention. A numerical η from the older convention does not have the same meaning and must be selected again. The candidate η grid itself was already in the older draft.

### 3. The unknown-linewidth penalty is specified differently

Old: a second-difference penalty on the five linewidth-knot values.

Current: a second-derivative penalty on G/Γ₀, using finite differences and quadrature on a normalised BFS axis, with residuals scaled by the water-column norm.

This changes/specifies the scale of the optimisation problem. It is more than a notation edit. The five-knot representation, alternating updates and ten starts were already planned.

## Equation changes: definitions, equivalent forms and evaluation choices

| Equation label | Exact nature of change |
|---|---|
| intro_brillouin_shift | Acoustic speed vL becomes VL. The BFS formula is unchanged. |
| linear_mixture | Uses explicitly broadened spectra R-tilde = h * R. The old paragraph already said the fitted component spectra include instrument broadening. Same additive model, clearer separation of intrinsic and recorded spectra. |
| glycerol_fraction | Same mass-fraction definition; punctuation/formatting only. |
| concentration_profile | Same two fields. d becomes ζ, radius subscript is upright, and the glycerol subscript is carried into the field names. |
| component_spectrum | Writes the same Lorentzian as R = Aφ and defines unit-area φ separately. Area and linewidth conventions are unchanged. |
| bfs_model | Same formula and coefficient value; a becomes αf. |
| linewidth_model | Same formula and coefficient value; b becomes αΓ. |
| intensity_model | Real denominator correction described above; δA becomes αA. |
| depth_integrated_spectrum | Limits 0…Lz become zf − Lz/2…zf + Lz/2. Baseline zf = Lz/2 gives the same old interval. This also enables the added centred-focus test. |
| fixed_mixture | Same two-component mixture and collection fraction. Coordinate and normalised-radius notation change. |
| signal_fraction | Same conversion from collected fraction to full-band signal fraction. |
| continuous_unmixing | Same probability-measure formulation, including the water atom. Dummy variable is consistently wg and the spatial notation changes. |
| noiseless_recorded | Same convolution plus background. |
| convolution | Same convolution integral; additionally defines h explicitly using φ and Γh. |
| noise_model | Same Poisson law. Replaces the position-indexed time ti by common t; adds the rescaled data yim = Nim/(tκ) and its expected bin integral. The old text already required a common count scale. |
| known_fit | Same NNLS loss. Dictionary is named Dfix; the sum is explicitly over the two components and the fitted solution fraction is identified as component 2. |
| mcr | Same nonnegative matrix-factorisation loss. |
| regularised_distribution | Real residual/penalty scaling change described above; normalised bin estimates are named ν-hat instead of π-hat. |
| unknown_family | Same signal-weighted mixture idea. Unit-area φ replaces ℓ; G and H become calligraphic; introduces the collection distribution ω and the explicit relation dξ = H dω. The old A = H statement is now conveyed through the measure relation and surrounding text. |
| proportion_mae | Replaces the unspecified sum/denominator N with an explicit evaluation set. Sample-intersecting positions become the primary average; whole-scan results remain secondary. |
| wasserstein | The old generic formula is written separately over concentration and BFS, with explicit limits and matching truth distributions. This creates one additional equation number; Wasserstein distance itself is unchanged. |
| spectral_nrmse | NRMSE is renamed rRMSE. The formula is identical. |
| longitudinal_modulus_decomposition | Same appendix modulus equation. |
| mechanical_moduli | Same appendix modulus equations; vL becomes VL and unit wording is formatted consistently. |

The actual old and current source for every displayed equation is included below in the equation audit.

## Numerical details clarified

- The outer frequency-bin edges now stop exactly at 4.50 and 8.50 GHz. Endpoint bins are therefore half-width (0.005 GHz); interior bins remain 0.01 GHz. The old table called all bin widths 0.01 GHz and did not specify the endpoint clipping.
- The bin integral, count scale and plotted spectral density are distinguished. J(wg) is defined in addition to J₀ so that retained-band signal fractions can be evaluated explicitly.
- The unknown-fixed-spectrum case normalises estimated spectra to unit area **within the retained band**, rescaling coefficients to preserve the fit. Its target signal fraction q is now given explicitly using J(wmax) and J₀, rather than leaving the normalisation convention to be specified later.
- The known-family concentration grid already had 51 values including water. It is now written explicitly as j = 0,…,50, distinguishing the single zero-concentration water mass from the 50 positive nodes. Nearest-node assignment was already planned.
- Unknown-family distribution error is explicitly signal-weighted. Linewidth error is assessed at the 49 interior BFS nodes, excluding the supplied endpoints.
- The old 0.05 error thresholds remain as interpretations of five percentage points or 5% of a coordinate range. However, the current text no longer explicitly promises to report the **fraction exceeding** those thresholds or to flag every exceedance. It promises full error distributions instead.
- The old failure definition used “relative tolerance of 10⁻⁸”; the current text says “constraint errors above 10⁻⁸” without specifying relative scaling. That qualifier was lost in shortening and is an unresolved specification detail.

## Extra proposed work added: these are choices, not essential corrections

These settings were added to make the plan more explicit. They are proposed synthetic tests and implementation choices, not measured parameters or completed results.

| Addition | Current choice |
|---|---|
| Harder spectral overlap | Smaller BFS coefficient 0.35 and wider water linewidth 0.40 GHz, separately and together |
| Centred collection | Extra zf = 0 comparison at Lz = 30 µm, besides the old interval beginning at z = 0 |
| Instrument shape | Gaussian response at FWHM 0.276 GHz, besides the already planned Lorentzian widths |
| Lateral averaging | Gaussian FWHM 2 µm, truncated at three standard deviations; the same weights define spectra and truth |
| Count-aware known-spectrum fit | Poisson negative-log-likelihood comparison, nonnegative coefficients, log floor 10⁻¹² counts |
| Single-peak fitting details | Three starting centres at band midpoint/quartiles; centre bounded to retained band; nonnegative area; linewidth 0.02–2.00 GHz |
| PCA endpoints | Extremes of the leading mean-centred component, followed by clipping of negative spectral values |
| VCA implementation | Project to K−1 dimensions; ten random seeds; validate on noiseless mixtures containing pure components |
| MCR-ALS initialisation | Ten starts: PCA, VCA and eight random positive estimates |
| Water-reference positions | Mean of 25 positions with x and y each in {−25,−24,−23,−22,−21} µm |
| Unknown-family discretisation | 51 BFS nodes and explicit 0.02–2.00 GHz linewidth bounds |
| Uncertainty intervals | 95% bootstrap intervals using 1000 resamples of paired whole noise realisations |

The old draft already included variations in maximum concentration, profile exponent, axial length, Lorentzian instrument width and area trend; the reserved radii 15/25 µm and exponents 0.75/1.5; a separate noiseless condition; 30 noisy repetitions; η selection on development data; and stopping after at most 1000 iterations or the stated relative-change criterion. Those are not new additions.

The convergence checks for lateral, depth, frequency and concentration sampling are now brought together under one 1% rule. Frequency and concentration refinement were already mentioned elsewhere in the old implementation section.

### Important change in the supplied endpoint information

Your older unknown-family method already mentioned “water-only and water-plus-10%-solution reference conditions”. The current version instead supplies **noiseless water and upper-concentration endpoint spectra**, fixing the BFS endpoints and endpoint linewidths. References were not invented as a new idea, but the current wording makes a stronger, idealised assumption about access to the endpoint spectra. That assumption is material to how difficult the inverse problem is.

The area-calibrated branch and the deliberately wrong constant-area comparison were already present. The current text expresses them more briefly.

## Irina's material concern

The new explanation makes the intended simulation scope explicit in both the aims and methodology:

- Water–glycerol provides a documented example of spectral features varying with concentration.
- Every homogeneous local solution contributes one positive-shift peak.
- Multiple contributions arise from different locations sampled by the collection volume.
- The sphere and concentration gradient are prescribed numerical fields, without a diffusion/stability model.
- The simulation tests recovery of known contributions and does not claim to reproduce tissue structure.

This develops an explanation already present in your older mathematical-test-object paragraph. It does not change to a different material or make glycerol and water separate pure-component peaks inside one uniform solution.

## Captions and images

All 15 captions have been revised. The exact old/new caption pairs appear below. Captions generally identify the panels and what is actually plotted, replace repeated caveats with clearer descriptions, and align photon budgets and fraction conventions with the methods.

The figure filenames in main.tex do not change. Figure 1's inclusion width changes from 0.72 to 1.0 times the line width. The other 14 inclusion widths remain 1.0.

**Image style history matters:** the earlier feedback pass regenerated all 15 assets with serif/Computer Modern styling. The latest corrections restore your supplied Arial font versions and original page dimensions for Figures 9, 12 and 13 only. The other 12 currently retain the earlier regenerated styling. This delivery does not silently change any fonts or images again.

| Figure | Current image changes relative to the earlier set |
|---|---|
| 1 | Compact side-by-side cell/probe and force–indentation panels; stiffness labels repositioned. Included wider in the paper. |
| 2 | Roomier apparatus labels, explicit incident optical frequency and frequency-shift labels, Lorentzian schematic peaks. Larger pixel grid and expanding guides retained. |
| 3 | Lorentzian examples, dotted peak centres, centre-to-centre BFS, clearly marked FWHM and repositioned labels. |
| 4 | PMMA terminology; removes the standalone title. |
| 5 | Stronger red collection box, clearer collection-volume terminology, wrapped titles and compact combined-spectrum legend. |
| 6 | Replaces the assumed 1000/10/1-second examples with reference budgets 10⁵/10⁴/10³, using the same benchmark bins. Grey expected traces and orange Poisson draws; raw-count axes. |
| 7 | Updated lettering/combined-spectrum wording; retains the single-Lorentzian least-squares example. |
| 8 | Clearer small labels and axis placement; distinguishes supplied references from inferred components. |
| 9 | Descriptive case names; fixed fractions 1/3 and 2/3. Panel C retains the original schematic bell-shaped distribution, restored on 18 October. It illustrates the recovery target, not the baseline concentration-bin masses. Panel D plots intrinsic linewidth against BFS rather than dual-axis quantities against glycerol percentage. Grey arrows sit between panels. The supplied Arial font and original size are preserved. |
| 10 | Stronger red collection cylinders, larger/readable 3-D ticks, regular depth ticks and coordinate labels separated from neighbouring panels. Caption says collection begins at the sample midplane. |
| 11 | Uses wg in the response equations, the fixed-reference intensity law and neutral curve colours. Caption now opens with what the panels show. |
| 12 | Moves the 3-D z label away from the intensity axis. Panel (c) is “Combined spectrum”, replacing “Their sum”. The supplied Arial font and original size are preserved. |
| 13 | Separates the 3-D z and depth labels; uses “Combined spectrum” consistently in title/legend; caption explains the concentration colour bar once. The supplied Arial font and original size are preserved. |
| 14 | Removes the repeated 3-D column: two rows × two columns now show line scans and selected mixed spectra. Common colour limits and a shared colour bar. |
| 15 | Matches Figure 14's compact layout; photon budget 10⁴ is explicitly over the retained band; exact-bin Poisson examples and matching scales. |

Figures 9, 12 and 13 preserve the dimensions of the individual PDFs you supplied: approximately 679.681 × 457.306 pt, 561.240 × 251.675 pt and 562.685 × 268.307 pt, respectively. The prior figure checks also verified preservation of their original font-family/size pairs and axes positions.

## References

The attached older source cites 11 keys, while the saved older bibliography contains nine entries. The current source cites 17 keys and the bibliography contains all 17.

- Added the two already-cited but missing entries: Discher et al. (2005) and Suresh (2007).
- Added six newly cited sources: Scarcelli and Yun (2008), Scarcelli et al. (2015), Kabakova et al. (2024), Bioucas-Dias et al. (2012), Lawson and Hanson (1995 reissue), and Keshava and Mustard (2002).
- Existing bibliography entries are retained; Brillouin/MCR capitalisation is protected where needed. The full bibliography text difference is included in the HTML comparison and patch.
- The glycerol reference's actual measurement wavelength/instrument correction are now described. It is explicitly distinguished from the synthetic 660 nm baseline.

## Text removed or compressed that may matter

- The long list of reference microscope objective sizes and interferometer details is shortened to the wavelength/resolution motivation.
- Repeated explanations of no pure-sample baseline rays, depth-vs-lateral contributions, non-uniqueness and calibration caveats are condensed. In particular, the explicit sentence about linewidth trading off against the spread of peak centres is removed, although the signal/material ambiguity is still explained.
- The detailed list of saved simulation metadata and separate analytic forward-model checks is compressed to recording the code commit, versions, settings, seeds and true spectra. The current text no longer itemises every saved array or every validation identity.
- The older explicit promise to examine variation across unknown-spectrum initialisations is shortened to specifying multiple starts; the current text does not explicitly promise to report that variation.
- The separate numerical/practical-failure exposition is shorter, with the threshold-reporting and relative-constraint qualifier changes noted above.
- Repeated statements that figures are not results are consolidated into the study-design/implementation explanations.

## What remains unfinished

The BFS/linewidth coefficients have **not** been fitted to the published glycerol measurements. They remain clearly labelled illustrative inputs. The forward-model figures are available; the recovery-method benchmarks, pilot tuning and reserved-sample comparisons remain planned work.

The current source has passed the earlier diagnostic TeX/BibTeX check, without overfull boxes. The built-in preview compiler was checked again for this delivery, but Windows blocked its executable with “Access is denied” before TeX ran. That is a preview-launch failure; it does not confirm or disprove final PDF layout. No replacement PDF was generated for this delivery.

## Download contents

- main.tex: the current source, identical to the editor file.
- BrillouinMicroscopy.bib: the matching 17-entry bibliography; use it with this main.tex.
- figures/: the 15 PDF assets used by main.tex.
- CHANGES_FROM_OLDER_DRAFT.md: this explained change record, including the complete equation/caption audit below.
- comparison/main_changes.html: every main-source and bibliography line, side by side.
- comparison/exact_changes.diff: the complete machine-generated text difference.
- comparison/older_main.tex and older_BrillouinMicroscopy.bib: preserved baselines.
- comparison/delivery_manifest.json: source hashes and checks.

Older Thesis.pdf exports and older ZIP packages are deliberately excluded.


## Complete displayed-equation audit

Each pair is copied directly from the sources, including notation and labels. Multi-line align blocks are kept together.

### 1. eq:intro_brillouin_shift

Older:

```latex
\begin{equation}
 f_{\mathrm B}=\frac{2nv_{\mathrm L}}{\lambda_0}\sin\left(\frac{\theta}{2}\right).
 \label{eq:intro_brillouin_shift}
\end{equation}
```

Current:

```latex
\begin{equation}
 f_{\mathrm B}=\frac{2nV_{\mathrm L}}{\lambda_0}\sin\left(\frac{\theta}{2}\right).
 \label{eq:intro_brillouin_shift}
\end{equation}
```

### 2. eq:linear_mixture

Older:

```latex
\begin{equation}
 y_i(f)=\sum_{k=1}^{K}a_{ik}R_k(f)+b_i(f)+\varepsilon_i(f),\qquad a_{ik}\geq0.
 \label{eq:linear_mixture}
\end{equation}
```

Current:

```latex
\begin{equation}
 y_i(f)=\sum_{k=1}^{K}a_{ik}\widetilde{R}_k(f)+b_i(f)+\varepsilon_i(f),\qquad a_{ik}\geq0.
 \label{eq:linear_mixture}
\end{equation}
```

### 3. eq:glycerol_fraction

Older:

```latex
\begin{equation}
 \wg=\frac{m_{\mathrm g}}{m_{\mathrm g}+m_{\mathrm w}},\qquad 100\wg=\text{glycerol percentage by mass},
 \label{eq:glycerol_fraction}
\end{equation}
```

Current:

```latex
\begin{equation}
 \wg=\frac{m_{\mathrm g}}{m_{\mathrm g}+m_{\mathrm w}},\qquad
 100\wg=\text{glycerol percentage by mass}.
 \label{eq:glycerol_fraction}
\end{equation}
```

### 4. eq:concentration_profile

Older:

```latex
\begin{align}
 w_{\mathrm{fixed}}(\mathbf r)&=
 \begin{cases}\wmax,&d<1,\\0,&d\geq1,\end{cases} \\
 w_{\mathrm{varying}}(\mathbf r)&=
 \begin{cases}\wmax(1-d)^\beta,&d<1,\\0,&d\geq1,\end{cases}
 \qquad \beta>0.
 \label{eq:concentration_profile}
\end{align}
```

Current:

```latex
\begin{align}
 w_{\mathrm{g,fixed}}(\mathbf r)&=\begin{cases}\wmax,&\zeta<1,\\0,&\zeta\geq1,\end{cases}\\
 w_{\mathrm{g,varying}}(\mathbf r)&=\begin{cases}\wmax(1-\zeta)^\beta,&\zeta<1,\\0,&\zeta\geq1,\end{cases}\qquad\beta>0.
 \label{eq:concentration_profile}
\end{align}
```

### 5. eq:component_spectrum

Older:

```latex
\begin{equation}
 R(f;\wg)=\frac{A(\wg)}{\pi}
 \frac{\Gamma(\wg)/2}{[f-f_{\mathrm B}(\wg)]^2+[\Gamma(\wg)/2]^2},
 \qquad \int_{-\infty}^{\infty}R(f;\wg)\,df=A(\wg).
 \label{eq:component_spectrum}
\end{equation}
```

Current:

```latex
\begin{equation}
 \begin{aligned}
 \phi(f;f_{\mathrm c},\Gamma)&=\frac{1}{\pi}
 \frac{\Gamma/2}{(f-f_{\mathrm c})^2+(\Gamma/2)^2},\\
 R(f;\wg)&=A(\wg)\phi\bigl(f;f_{\mathrm B}(\wg),\Gamma(\wg)\bigr),\\
 \int_{-\infty}^{\infty}R(f;\wg)\dd f&=A(\wg).
 \end{aligned}
 \label{eq:component_spectrum}
\end{equation}
```

### 6. eq:bfs_model, eq:linewidth_model, eq:intensity_model

Older:

```latex
\begin{align}
 f_{\mathrm B}(\wg)&=f_{\mathrm B0}(1+a\wg), \label{eq:bfs_model}\\
 \Gamma(\wg)&=\Gamma_0\left[\left(\frac{f_{\mathrm B}(\wg)}{f_{\mathrm B0}}\right)^2+b\wg\right], \label{eq:linewidth_model}\\
 A(\wg)&=A_0\left(1+\delta_A\frac{\wg}{\wmax}\right). \label{eq:intensity_model}
\end{align}
```

Current:

```latex
\begin{align}
 f_{\mathrm B}(\wg)&=f_{\mathrm{B0}}(1+\alpha_f\wg),\label{eq:bfs_model}\\
 \Gamma(\wg)&=\Gamma_0\left[\left(\frac{f_{\mathrm B}(\wg)}{f_{\mathrm{B0}}}\right)^2+\alpha_\Gamma\wg\right],\label{eq:linewidth_model}\\
 A(\wg)&=A_0\left(1+\alpha_A\frac{\wg}{w_{\mathrm{ref}}}\right),\qquad w_{\mathrm{ref}}=0.10.
 \label{eq:intensity_model}
\end{align}
```

### 7. eq:depth_integrated_spectrum

Older:

```latex
\begin{equation}
 s_i(f)=\int_{0}^{L_z}W_i(z)R(f;w_i(z))\,dz,
 \qquad W_i(z)\geq0,\qquad \int W_i(z)\,dz=1.
 \label{eq:depth_integrated_spectrum}
\end{equation}
```

Current:

```latex
\begin{equation}
 s_i(f)=\int_{z_{\mathrm f}-L_z/2}^{z_{\mathrm f}+L_z/2}
 W_i(z)R\bigl(f;w_{\mathrm{g},i}(z)\bigr)\dd z,
 \qquad W_i(z)\geq0,\qquad\int W_i(z)\dd z=1.
 \label{eq:depth_integrated_spectrum}
\end{equation}
```

### 8. eq:fixed_mixture

Older:

```latex
\begin{equation}
 s_i(f)=(1-p_i)R(f;0)+p_iR(f;\wmax),\qquad
 p_i=\int W_i(z)\,\mathbf{1}_{\{d(x_i,y_i,z)<1\}}\,dz.
 \label{eq:fixed_mixture}
\end{equation}
```

Current:

```latex
\begin{equation}
 s_i(f)=(1-p_i)R(f;0)+p_iR(f;\wmax),\qquad
 p_i=\int W_i(z)\mathbf1_{\{\zeta(\mathbf r_i,z)<1\}}\dd z.
 \label{eq:fixed_mixture}
\end{equation}
```

### 9. eq:signal_fraction

Older:

```latex
\begin{equation}
 p_i^{\mathrm{signal}}=
 \frac{p_i A(\wmax)}{(1-p_i)A(0)+p_i A(\wmax)}.
 \label{eq:signal_fraction}
\end{equation}
```

Current:

```latex
\begin{equation}
 p_i^{\mathrm{signal}}=\frac{p_iA(\wmax)}{(1-p_i)A_0+p_iA(\wmax)}.
 \label{eq:signal_fraction}
\end{equation}
```

### 10. eq:continuous_unmixing

Older:

```latex
\begin{equation}
 s_i(f)=\int_{[0,\wmax]}R(f;w)\,d\nu_i(w),\qquad
 \nu_i(B)=\int W_i(z)\,\mathbf{1}_{\{w_i(z)\in B\}}\,dz.
 \label{eq:continuous_unmixing}
\end{equation}
```

Current:

```latex
\begin{equation}
 s_i(f)=\int_{[0,\wmax]}R(f;\wg)\dd\nu_i(\wg),\qquad
 \nu_i(B)=\int W_i(z)\mathbf1_{\{w_{\mathrm{g},i}(z)\in B\}}\dd z.
 \label{eq:continuous_unmixing}
\end{equation}
```

### 11. eq:noiseless_recorded, eq:convolution

Older:

```latex
\begin{align}
 y_i^{(0)}(f)&=(h*s_i)(f)+b_i(f), \label{eq:noiseless_recorded}\\
 (h*s_i)(f)&=\int_{-\infty}^{\infty}h(f-f')s_i(f')\,df',\qquad
 \int h(f)\,df=1. \label{eq:convolution}
\end{align}
```

Current:

```latex
\begin{align}
 y_i^{(0)}(f)&=(h*s_i)(f)+b_i(f),\label{eq:noiseless_recorded}\\
 (h*s_i)(f)&=\int_{-\infty}^{\infty}h(f-f')s_i(f')\dd f',\qquad
 h(f)=\phi(f;0,\Gamma_h),\qquad\int h(f)\dd f=1.
 \label{eq:convolution}
\end{align}
```

### 12. eq:noise_model

Older:

```latex
\begin{equation}
 N_{im}\sim\operatorname{Poisson}(\mu_{im}),\qquad
 \mu_{im}=t_i\kappa\int_{\Delta f_m}y_i^{(0)}(f)\,df.
 \label{eq:noise_model}
\end{equation}
```

Current:

```latex
\begin{equation}
 N_{im}\sim\operatorname{Poisson}(\mu_{im}),\qquad
 \mu_{im}=t\kappa\int_{\Delta f_m}y_i^{(0)}(f)\dd f,\qquad
 y_{im}=\frac{N_{im}}{t\kappa},\qquad
 y_{im}^{(0)}=\int_{\Delta f_m}y_i^{(0)}(f)\dd f.
 \label{eq:noise_model}
\end{equation}
```

### 13. eq:known_fit

Older:

```latex
\begin{equation}
 \widehat{\mathbf a}_i=\underset{\mathbf a_i\geq0}{\operatorname{argmin}}
 \|\mathbf y_i-\mathbf D\mathbf a_i\|_2^2,
 \qquad \widehat p_{ik}=\frac{\widehat a_{ik}}{\sum_\ell\widehat a_{i\ell}}.
 \label{eq:known_fit}
\end{equation}
```

Current:

```latex
\begin{equation}
 \widehat{\mathbf a}_i=\underset{\mathbf a_i\geq0}{\operatorname{argmin}}
 \|\mathbf y_i-\mathbf D_{\mathrm{fix}}\mathbf a_i\|_2^2,\qquad
 \widehat p_{ik}=\frac{\widehat a_{ik}}{\sum_{k'=1}^{2}\widehat a_{ik'}},\qquad
 \widehat p_i=\widehat p_{i2}.
 \label{eq:known_fit}
\end{equation}
```

### 14. eq:mcr

Older:

```latex
\begin{equation}
 \min_{\mathbf C\geq0,\,\mathbf S\geq0}
 \|\mathbf Y-\mathbf C\mathbf S^{\mathsf T}\|_{\mathrm F}^{2},
 \label{eq:mcr}
\end{equation}
```

Current:

```latex
\begin{equation}
 \min_{\mathbf C\geq0,\mathbf S\geq0}
 \|\mathbf Y-\mathbf C\mathbf S^{\mathsf T}\|_{\mathrm F}^{2}.
 \label{eq:mcr}
\end{equation}
```

### 15. eq:regularised_distribution

Older:

```latex
\begin{equation}
 \widehat{\mathbf u}_i=\underset{\mathbf u_i\geq0}{\operatorname{argmin}}
 \left[\|\mathbf y_i-\mathbf D\mathbf u_i\|_2^2+
 \eta\|\mathbf L\mathbf u_{i,+}\|_2^2\right],\qquad
 \widehat\pi_{ij}=\frac{\widehat u_{ij}}{\sum_\ell\widehat u_{i\ell}}.
 \label{eq:regularised_distribution}
\end{equation}
```

Current:

```latex
\begin{equation}
 \widehat{\mathbf u}_i=\underset{\mathbf u_i\geq0}{\operatorname{argmin}}
 \left[\frac{\|\mathbf y_i-\mathbf D_{\mathrm{grid}}\mathbf u_i\|_2^2}
 {\|\mathbf D_{\mathrm{grid},0}\|_2^2}
 +\eta\|\mathbf L_w\mathbf u_{i,+}\|_2^2\right],\qquad
 \widehat\nu_{ij}=\frac{\widehat u_{ij}}{\sum_{j'=0}^{50}\widehat u_{ij'}}.
 \label{eq:regularised_distribution}
\end{equation}
```

### 16. eq:unknown_family

Older:

```latex
\begin{equation}
 \Gamma=G(f_{\mathrm B}),\qquad A=H(f_{\mathrm B}),\qquad
 s_i(f)=\int \ell(f;u,G(u))\,d\xi_i(u),
 \label{eq:unknown_family}
\end{equation}
```

Current:

```latex
\begin{equation}
 \Gamma=\mathcal G(f_{\mathrm B}),\qquad
 s_i(f)=\int\phi\bigl(f;f_{\mathrm B},\mathcal G(f_{\mathrm B})\bigr)\dd\xi_i(f_{\mathrm B}),\qquad
 \dd\xi_i(f_{\mathrm B})=\mathcal H(f_{\mathrm B})\dd\omega_i(f_{\mathrm B}).
 \label{eq:unknown_family}
\end{equation}
```

### 17. eq:proportion_mae

Older:

```latex
\begin{equation}
 \mathrm{MAE}_{p}=\frac{1}{N}\sum_i|\widehat p_i-p_i|.
 \label{eq:proportion_mae}
\end{equation}
```

Current:

```latex
\begin{equation}
 \mathrm{MAE}_p=\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}|\widehat p_i-p_i|.
 \label{eq:proportion_mae}
\end{equation}
```

### 18. eq:wasserstein

Older:

```latex
\begin{equation}
 W_{1,i}=\int|\widehat F_i(x)-F_i(x)|\,dx.
 \label{eq:wasserstein}
\end{equation}
```

Current:

```latex
\begin{align}
 \mathcal W_{1,i}^{w}&=\int_0^{\wmax}
 |\widehat F_i^w(\wg)-F_i^w(\wg)|\dd\wg,\\
 \mathcal W_{1,i}^{\mathrm B}&=\int_{f_{\mathrm{B,min}}}^{f_{\mathrm{B,max}}}
 |\widehat F_i^{\mathrm B}(f_{\mathrm B})-F_i^{\mathrm B}(f_{\mathrm B})|\dd f_{\mathrm B}.
 \label{eq:wasserstein}
\end{align}
```

### 19. eq:spectral_nrmse

Older:

```latex
\begin{equation}
 \mathrm{NRMSE}_i=
 \sqrt{\frac{\sum_m[\widehat y_{im}-y^{(0)}_{im}]^2}
 {\sum_m[y^{(0)}_{im}]^2}}.
 \label{eq:spectral_nrmse}
\end{equation}
```

Current:

```latex
\begin{equation}
 \mathrm{rRMSE}_i=\sqrt{\frac{\sum_m[\widehat y_{im}-y_{im}^{(0)}]^2}
 {\sum_m[y_{im}^{(0)}]^2}}.
 \label{eq:spectral_nrmse}
\end{equation}
```

### 20. eq:longitudinal_modulus_decomposition

Older:

```latex
\begin{equation}
 M'=K'+\frac{4}{3}G'.
 \label{eq:longitudinal_modulus_decomposition}
\end{equation}
```

Current:

```latex
\begin{equation}
 M'=K'+\frac{4}{3}G'.
 \label{eq:longitudinal_modulus_decomposition}
\end{equation}
```

### 21. eq:mechanical_moduli

Older:

```latex
\begin{equation}
 M'\simeq\rho v_{\mathrm L}^{\,2}
 =\rho\left[\frac{\lambda_0 f_{\mathrm B}}{2n\sin(\theta/2)}\right]^2,
 \qquad \frac{M''}{M'}\simeq\frac{\Gamma}{f_{\mathrm B}}.
 \label{eq:mechanical_moduli}
\end{equation}
```

Current:

```latex
\begin{equation}
 M'\simeq\rho V_{\mathrm L}^{\,2}
 =\rho\left[\frac{\lambda_0 f_{\mathrm B}}{2n\sin(\theta/2)}\right]^2,
 \qquad \frac{M''}{M'}\simeq\frac{\Gamma}{f_{\mathrm B}}.
 \label{eq:mechanical_moduli}
\end{equation}
```


## Complete caption audit

Captions are copied directly from the sources. LaTeX commands are retained so no wording or mathematical detail is hidden.

### Figure 1: 01_cell_mechanics.pdf

Older:

```latex
Similar appearances can conceal different mechanical responses. An AFM probe applies force to a cell; at the same force, the higher stiffness response has less indentation.
```

Current:

```latex
Cell deformation under a common applied force. (a) An AFM probe presses into a cell. (b) At the same force $F$, the higher-stiffness response has smaller indentation, $\delta_{\mathrm H}<\delta_{\mathrm L}$.
```

### Figure 2: 02_brillouin_measurement.pdf

Older:

```latex
How Brillouin microscopy produces a spectral image. (a) Incident light is focused into a sample and backscattered light is directed to a separate spectrometer and detector. Acoustic-wave motifs are symbolic. (b) Each spatial scan position contains a spectrum. (c) Expanding guides connect one pixel to its cartoon photon-count spectrum. The peaks, geometry and counts are schematic; the angle is ideal backscattering.
```

Current:

```latex
Spectral image formation in Brillouin microscopy. (a) Light at optical frequency $f_{\mathrm{inc}}$ is focused into the sample and backscattered light reaches a separate spectrometer and detector. (b) A spatial scan records a spectrum at each position. (c) The selected pixel contains incident-light, Stokes and anti-Stokes peaks at frequency shifts $0$, $-f_{\mathrm B}$ and $+f_{\mathrm B}$.
```

### Figure 3: 03_spectral_features.pdf

Older:

```latex
Reading a Brillouin spectrum. (a) BFS is the separation from the incident-light frequency, and FWHM is the width at half peak height. The lower panels distinguish (b) a centre shift from (c) increased linewidth. Curves are schematic; the simulation models only the positive-shift Brillouin band.
```

Current:

```latex
Brillouin spectral features. (a) BFS is the separation between peak centres; FWHM $\Gamma$ is the width at half maximum. The drawn peaks are Lorentzian. (b) A centre shift changes BFS. (c) Broadening changes linewidth at constant integrated intensity.
```

### Figure 4: 04_material_contrast.pdf

Older:

```latex
Different materials can have separated Brillouin responses. These schematic water and Plexiglass peaks introduce material contrast; they are not the water--glycerol simulation calibration.
```

Current:

```latex
Material contrast in water and PMMA. Separated peaks illustrate how component spectra can be distinguished when their BFS values differ sufficiently.
```

### Figure 5: 05_collection_volume.pdf

Older:

```latex
A finite collection volume can include sample and surrounding medium. Their weighted local responses combine into one measured spectrum. The colours identify contributions, not separate reconstructed depth positions.
```

Current:

```latex
Mixed contributions within one collection volume. (a) The outlined volume contains sample and medium. (b) Their weighted component spectra. (c) The combined spectrum on the same frequency and intensity scales.
```

### Figure 6: 06_exposure_and_shot_noise.pdf

Older:

```latex
Shorter exposure increases relative photon shot noise. Connected orange lines show independent Poisson-count realisations, with faint blue expected traces. The underlying mixed response and photon rate are fixed; only exposure changes. Raw-count axes change scale between panels. The illustrative scale is 1000 expected water-reference photons per second in the displayed 5.25--7.35 GHz band; these times are not an experimental calibration.
```

Current:

```latex
Photon budget and shot noise. Orange traces are Poisson realisations and grey traces their noiseless expectations for the fixed sample. The pure-water reference budgets are $P=10^5$, $10^4$ and $10^3$ in the retained band, using the bin model of Equation~\eqref{eq:noise_model}. Vertical axes show raw photon counts.
```

### Figure 7: 07_single_peak_fit_limit.pdf

Older:

```latex
A single peak misses the shape of this chosen two-component mixture. (a) The true components and their sum. (b) The same sum and its best single-Lorentzian least-squares fit. The panels have common frequency and intensity scales. Stronger overlap can make the mismatch less apparent.
```

Current:

```latex
Single-peak fitting of a two-component mixture. (a) Component spectra and their combined spectrum. (b) The same combined spectrum and a least-squares single-Lorentzian fit. Both panels use common frequency and intensity scales.
```

### Figure 8: 08_spectral_unmixing.pdf

Older:

```latex
Known and unknown spectral unmixing. Known references permit fitting of contributions; unknown references must be estimated from variation across mixed measurements. Each small spectrum is one continuous curve in the Brillouin band. Recovered outputs are illustrative, not benchmark results.
```

Current:

```latex
Unmixing with known and unknown component spectra. (a) Supplied reference spectra permit fitting of normalised fractions. (b) Multiple mixed spectra provide the variation needed to estimate component spectra and their fractions.
```

### Figure 9: 09_study_design.pdf

Older:

```latex
Four simulation configurations and their recovery targets. (1) Known fixed spectra are used to estimate mixture weights. (2) Unknown fixed spectra require estimation of component spectra and weights. (3) Known concentration-dependent spectra are used to estimate a concentration distribution. (4) Unknown varying spectra require estimation of contributions and spectral relationships. The final panel illustrates the prescribed BFS (blue dashed line, left axis) and intrinsic linewidth (orange solid line, right axis) versus glycerol mass percentage. All example outputs are schematic, not recovery results. Unknown-spectrum weights require intensity calibration to represent collection fractions; the final configuration does not imply uniquely recoverable absolute concentration.
```

Current:

```latex
Inputs and recovery targets for the four cases. The fixed sample is analysed with supplied spectra or with spectra estimated from mixed measurements. The concentration distribution is schematic and illustrates the recovery target. When the local spectra are unknown, water and upper-solution endpoint spectra are supplied; the final panel shows the intrinsic-linewidth relationship versus BFS to be recovered.
```

### Figure 10: 10_sample_models.pdf

Older:

```latex
Three-dimensional sample definitions. A uniform 10\% solution and a radial gradient are surrounded by water. The gradient cutaway reveals prescribed internal concentration. The red array shows collection regions at representative lateral scan positions, each spanning the 30 $\mu\mathrm m$ axial collection range, from the sample centre at $z=0$ to $z=30\,\mu\mathrm m$. These are simulation test objects, not reconstructed volumes or diffusion predictions.
```

Current:

```latex
Fixed and varying sample fields in surrounding water. Red cylinders mark collection volumes at representative scan positions. Each axial collection interval extends from the sample midplane ($z=0$) to $z=\qty{30}{\micro\metre}$, with focus $z_{\mathrm f}=\qty{15}{\micro\metre}$. The varying sample has a radial concentration gradient.
```

### Figure 11: 11_concentration_relationships.pdf

Older:

```latex
Here $w$ denotes glycerol mass fraction (percentage divided by 100). Prescribed response laws over the initial 0--10\% glycerol mass range. BFS and intrinsic linewidth follow the provisional model. Integrated intensity is chosen to increase linearly by 20\% across this range. The curves are simulation inputs, not measured or fitted calibrations; integrated intensity means peak area.
```

Current:

```latex
Prescribed concentration dependence of the component spectra. (a) BFS, (b) intrinsic linewidth and (c) integrated intensity follow Equations~\eqref{eq:bfs_model}--\eqref{eq:intensity_model}. The fixed reference concentration $w_{\mathrm{ref}}=0.10$ gives $A/A_0=1.20$ at 10\% glycerol.
```

### Figure 12: 12_fixed_mixture_formation.pdf

Older:

```latex
Formation of a fixed-component spectrum along one collection ray. The central ray contains collection fractions 33.3\% water and 66.7\% homogeneous 10\% solution (full-band signal fractions 29.4\% and 70.6\%, respectively). Coloured curves include these weights and the concentration-dependent response areas; their exact sum is black. The spectra include the assumed instrument response. Intensity denotes spectral density in the common arbitrary scale.
```

Current:

```latex
Combined-spectrum formation in the fixed sample. (a) A central collection volume contains $1/3$ water and $2/3$ homogeneous solution by collection weight. (b) Weighted component spectra. (c) Their combined spectrum. The corresponding full-band signal fractions are 29.4\% water and 70.6\% solution.
```

### Figure 13: 13_gradient_mixture_formation.pdf

Older:

```latex
Formation of a varying-concentration spectrum. (a) Coloured points lie at different depths within one collection region. (b) Local spectra are individually scaled and offset vertically for visibility. (c) Water and concentration-bin contributions are depth-weighted and summed exactly. The colour bar identifies glycerol concentration for the coloured contributions. Colour represents glycerol mass fraction. Collection begins at the centre and traverses one side of the gradient. These are depth contributions to one pixel, not different lateral pixels.
```

Current:

```latex
Combined-spectrum formation in the varying sample. (a) Selected depth positions within one collection volume. (b) Local spectra, individually scaled and vertically offset to show their shapes. (c) Weighted contributions and the black combined spectrum. The colour bar gives glycerol percentage by mass.
```

### Figure 14: 14_noiseless_dataset.pdf

Older:

```latex
Noiseless forward-model datasets. Left: three-dimensional sample models and red collection indicators. Middle: the central lateral spectral line, displayed over 5.65--7.00 GHz; the complete simulated band is 4.50--8.50 GHz. The white line selects $x=y=0$. Right: contributions to that single pixel and their exact black sum. Intensity denotes spectral density on the common arbitrary scale. Both rows use common scales. Instrument broadening and the chosen increasing-intensity law are included; random noise and additive background are absent.
```

Current:

```latex
Noiseless spectral datasets. (a,c) Central line scans of the fixed and varying samples; the dashed line selects the central pixel. (b,d) Its component contributions and black combined spectrum. The display window is 5.65--\qty{7.00}{\giga\hertz}; stored spectra retain $\mathcal B$. Both rows include instrument broadening and use common colour limits.
```

### Figure 15: 15_noisy_dataset.pdf

Older:

```latex
Poisson-noise examples corresponding to Figure~\ref{fig:noiseless_dataset}, with the same geometry and heatmap colour limits. The reference budget is 10000 expected water photons per pixel in the full band. Counts are converted to spectral density (labelled intensity) using the known count scale and bin width for display. Orange traces show noisy pixels and faint curves show their expectations. These are simulations, not experimental measurements.
```

Current:

```latex
Noisy spectral datasets at photon budget $P=10^4$. (a,c) Line scans of the fixed and varying samples. (b,d) Selected-pixel traces, with orange noisy realisations and grey noiseless expectations. Counts follow Equation~\eqref{eq:noise_model} over the retained band and are divided by count scale and bin width for intensity display. Heatmap limits match Figure~\ref{fig:noiseless_dataset}.
```
