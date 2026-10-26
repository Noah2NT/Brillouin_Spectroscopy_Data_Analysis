"""Lorentzian fit demo: pick a sample, click a pixel, see raw vs fitted spectrum.

Run:  python analysis/misc/lorentzian_demo.py   then open http://localhost:8000
"""
import io, json, sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import brillouin_preprocessing as bp
import brillouin_fitting as bf

SAMPLES = sorted((ROOT / "data/csv").glob("*.csv"))
MAX_SHIFT_GHZ = 9.0


@lru_cache(maxsize=4)
def load(i):
    spectra, freq, df = bp.load_brillouin_csv(SAMPLES[i])
    folded, shift = bp.fold_brillouin_spectrum(spectra, freq)
    keep = shift <= MAX_SHIFT_GHZ
    folded, shift = folded[:, keep], shift[keep]
    x_vals, y_vals = bp.build_coordinate_arrays(df)
    xi = np.searchsorted(x_vals, df["X"].to_numpy())
    yi = np.searchsorted(y_vals, df["Y"].to_numpy())
    # Min-max normalise each pixel to [0, 1] so every curve fills the same axis.
    lo = folded.min(axis=1); scale = np.maximum(folded.max(axis=1) - lo, 1.0)
    counts = folded; folded = bp.normalize_spectra(folded)
    curves, params = bf.fit_all_spectra(shift, folded)
    # ponytail: shot-noise gain alpha (var = alpha * signal), in raw counts, from the single-fit
    # residuals over the whole map; use two_material_solve.fit_alpha on a bg mask if it matters
    resid = (counts - lo[:, None]) - curves * scale[:, None]
    alpha = float((resid ** 2).sum() / np.clip(curves * scale[:, None], 1e-9, None).sum())
    # normalised sigma = sqrt(alpha * y_norm / scale); axis limit = 2x the largest band in the sample
    lim = 2 * 1.96 * np.sqrt(alpha / np.percentile(scale, 1))
    return dict(shift=shift, folded=folded, curves=curves, params=params, alpha=alpha, scale=scale,
                lim=lim, xi=xi, yi=yi, nx=len(x_vals), ny=len(y_vals))


def lorentzian2(x, a1, m1, g1, a2, m2, g2, c):
    return bf.lorentzian(x, a1, m1, g1, 0) + bf.lorentzian(x, a2, m2, g2, c)


def fit_double(shift, y, p1):
    """Two Lorentzians + shared baseline, seeded from the single fit."""
    from scipy.optimize import curve_fit
    a, m, g, c = p1
    lo, hi = float(shift.min()), float(shift.max())
    p0 = [a, m, g, 0.2 * a, float(np.clip(m + 0.5, lo, hi)), g, c]
    try:
        popt, _ = curve_fit(lorentzian2, shift, y, p0=p0, maxfev=4000,
                            bounds=([0, lo, 1e-4] * 2 + [-np.inf], [np.inf, hi, hi - lo] * 2 + [np.inf]))
    except (RuntimeError, ValueError):
        popt = np.array(p0)
    return lorentzian2(shift, *popt), popt


@lru_cache(maxsize=None)
def endmember(i, shift):
    """Background endmember from outputs/background_endmember, interpolated onto our axis."""
    import pandas as pd
    em = pd.read_csv(ROOT / "outputs/background_endmember" / f"{SAMPLES[i].stem}_background_endmember.csv")
    return np.interp(np.asarray(shift), em.shift_ghz, em.endmember_norm)


def fit_bg(shift, y, p_ols, em):
    """Lorentzian + x * background endmember, Poisson NLL — same recipe as
    lorentzian_plus_background_fit.fit_all for one pixel (alpha only rescales the loss)."""
    sys.path.insert(0, str(ROOT / "analysis/pca_vca"))
    import lorentzian_plus_background_fit as lpb
    lo, hi, span = float(shift.min()), float(shift.max()), float(shift.max() - shift.min())
    b1 = [(0, None), (lo, hi), (1e-3, span), (0, None)]; b2 = b1 + [(0, None)]
    tot = max(y.sum(), 1.0); ys = y / tot
    p0 = np.array([max(p_ols[0], 0) / tot, np.clip(p_ols[1], lo, hi), np.clip(p_ols[2], 1e-3, span), max(p_ols[3], 0) / tot])
    p1, _ = lpb.fit_poisson(ys, shift, 1.0, p0, b1)
    starts = [np.append(p1, 0.0), np.array([0.2 * p1[0], p1[1], p1[2], 0.0, 0.8])]
    p2, _ = min((lpb.fit_poisson(ys, shift, 1.0, s, b2, em) for s in starts), key=lambda t: t[1])
    p2 = p2 * np.array([tot, 1, 1, tot, tot])
    return lpb.model(shift, p2, em), p2


def fit_png(i, p, model="single", error=False):
    d = load(i); x, y = d["shift"], d["folded"][p]
    amp, mean, gamma, c = d["params"][p]
    if model == "bg":
        em = endmember(i, tuple(x))
        curve, q = fit_bg(x, y, d["params"][p], em)
        info = f"shift={q[1]:.3f} GHz  FWHM={2*q[2]:.3f} GHz  bg share x={q[4]:.2f}"
        label = "Lorentzian + background EM"
    elif model == "double":
        curve, q = fit_double(x, y, d["params"][p])
        info = f"peaks {q[1]:.3f} / {q[4]:.3f} GHz  FWHM {2*q[2]:.3f} / {2*q[5]:.3f} GHz"
        label = "double Lorentzian fit"
    else:
        curve, label = d["curves"][p], "Lorentzian fit"
        info = f"shift={mean:.3f} GHz  FWHM={2*gamma:.3f} GHz  peak={d["scale"][p]:.0f} counts"
    if error:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 5.5), sharex=True, height_ratios=[2, 1])
        res = y - curve
        band = 1.96 * np.sqrt(d["alpha"] * np.clip(curve, 0, None) / d["scale"][p])   # shot noise in normalised units
        lim = d["lim"]
        ax2.fill_between(x, -band, band, color="r", alpha=0.2,
                         label=f"expected shot noise (±1.96σ, α={d['alpha']:.2f}, peak={d['scale'][p]:.0f} counts)")
        ax2.axhline(0, color="r", lw=1)
        ax2.plot(x, res, "k-", lw=1, label="raw − fit")
        ax2.set(xlabel="Brillouin shift (GHz)", ylabel="raw − fit", ylim=(-lim, lim)); ax2.legend(fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, ".", ms=3, label="raw (folded)")
    ax.plot(x, curve, "r-", label=label)
    if model == "double":
        ax.plot(x, bf.lorentzian(x, q[0], q[1], q[2], q[6]), "g--", lw=1, label="peak 1")
        ax.plot(x, bf.lorentzian(x, q[3], q[4], q[5], q[6]), "m--", lw=1, label="peak 2")
    if model == "bg":
        ax.plot(x, bf.lorentzian(x, *q[:4]), "g--", lw=1, label="Lorentzian")
        ax.plot(x, q[4] * em, "m--", lw=1, label="x · background EM")
    ax.set(xlabel=None if error else "Brillouin shift (GHz)", ylabel="Intensity (normalised)", ylim=(-0.1, 1.1),
           title=f"pixel {p}  (x={d['xi'][p]}, y={d['yi'][p]})\n{info}")
    ax.title.set_fontsize(9); ax.legend(fontsize=8); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return buf.getvalue()


INDEX = "<h2>Lorentzian fit demo</h2>" + "".join(
    f'<p><a href="/grid?sample={i}" target="_blank">{s.name}</a></p>' for i, s in enumerate(SAMPLES))

GRID = """<!doctype html><title>Lorentzian demo</title>
<style>body{font-family:sans-serif;display:flex;gap:20px}
#g{display:grid;gap:1px}#g div{width:14px;height:14px;cursor:pointer}
#g div:hover{outline:2px solid #000}</style>
<div><h3 id=t>fitting all pixels...</h3><div id=g></div></div>
<div><label><input type=radio name=m value=single checked> single Lorentzian</label>
<label><input type=radio name=m value=double> double Lorentzian</label>
<label><input type=radio name=m value=bg> background EM + Lorentzian</label>
<label><input type=checkbox id=e> show error</label><br><img id=im width=600></div>
<script>
const s=new URLSearchParams(location.search).get('sample');let cur=null;
function show(){if(cur===null)return;const m=document.querySelector('input[name=m]:checked').value;
  document.getElementById('im').src=`/fit?sample=${s}&p=${cur}&model=${m}&error=${+document.getElementById('e').checked}`;}
document.querySelectorAll('input').forEach(i=>i.onchange=show);
fetch('/map?sample='+s).then(r=>r.json()).then(m=>{
  const g=document.getElementById('g');g.style.gridTemplateColumns=`repeat(${m.nx},14px)`;
  const v=m.shift.filter(x=>x!==null),lo=Math.min(...v),hi=Math.max(...v);
  document.getElementById('t').textContent=`${m.name}  (shift ${lo.toFixed(2)}-${hi.toFixed(2)} GHz)`;
  m.grid.forEach(p=>{const d=document.createElement('div');
    if(p===null){d.style.background='#ddd';}else{
      const t=(m.shift[p]-lo)/(hi-lo||1);d.style.background=`hsl(${240-240*t},80%,50%)`;
      d.onclick=()=>{cur=p;show();};}
    g.appendChild(d);});});
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        i = int(q.get("sample", [0])[0])
        if u.path == "/":
            body, ctype = INDEX.encode(), "text/html"
        elif u.path == "/grid":
            body, ctype = GRID.encode(), "text/html"
        elif u.path == "/map":
            d = load(i)
            grid = np.full(d["ny"] * d["nx"], -1); grid[d["yi"] * d["nx"] + d["xi"]] = np.arange(len(d["xi"]))
            body = json.dumps(dict(name=SAMPLES[i].name, nx=d["nx"], ny=d["ny"],
                                   grid=[None if g < 0 else int(g) for g in grid],
                                   shift=d["params"][:, 1].tolist())).encode()
            ctype = "application/json"
        elif u.path == "/fit":
            body, ctype = fit_png(i, int(q["p"][0]), q.get("model", ["single"])[0], q.get("error", ["0"])[0] == "1"), "image/png"
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("http://localhost:8000"); HTTPServer(("", 8000), H).serve_forever()
