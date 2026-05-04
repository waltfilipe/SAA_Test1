import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
import math
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict

# =============================================================================
# Page + Style
# =============================================================================
st.set_page_config(layout="wide", page_title="Pass Map Dashboard")

st.markdown("""
<style>
.small-metric{padding:6px 8px;}
.small-metric .label{font-size:12px;color:#ffffff;margin-bottom:3px;opacity:.95;}
.small-metric .value{font-size:18px;font-weight:600;color:#ffffff;}
.small-metric .delta{font-size:11px;color:#e6e6e6;margin-top:4px;}
.stats-section-title{font-size:14px;font-weight:600;margin-bottom:6px;color:#ffffff;}
.streamlit-expanderHeader{color:#ffffff!important;}
.streamlit-expander{background:rgba(255,255,255,.02);}
.filter-panel{
  background:linear-gradient(168deg,rgba(30,39,56,.92) 0%,rgba(22,28,40,.97) 100%);
  border:1px solid rgba(255,255,255,.08);border-radius:14px;
  padding:24px 18px 20px 18px;
  box-shadow:0 4px 24px rgba(0,0,0,.25),0 1px 4px rgba(0,0,0,.12);
  backdrop-filter:blur(6px);}
.filter-panel h3{font-size:15px;color:#c8d6e5;letter-spacing:.5px;margin-bottom:8px;}
.filter-panel .filter-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:14px 0;}
.stSubheader{color:#ffffff!important;}
</style>
""", unsafe_allow_html=True)

def small_metric(label: str, value: str, delta: str | None = None):
    html = f'<div class="small-metric"><div class="label">{label}</div><div class="value">{value}</div>'
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

st.title("Pass Map Dashboard")

# =============================================================================
# Constants
# =============================================================================
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67

NX, NY = 16, 12
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

DELTA_THR = 0.05

COLOR_FAIL_LIGHT = "#F2A7A7"
COLOR_ALL_PASS_LIGHT = "#E4E7EE"

PRESSURE_BONUS = {"0-0": 0.00, "1-0": 0.06, "0-1": 0.06, "1-1": 0.12}
PRESSURE_LABEL = {
    "0-0": "No pressure",
    "1-0": "Pressure at origin",
    "0-1": "Pressure at destination",
    "1-1": "Pressure at both",
}

# =============================================================================
# xT Grid
# =============================================================================
@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24,
                    goal_width=11.0, penalty_depth=18.5, penalty_width=45.32,
                    prox_w=0.50, central_w=0.50,
                    internal_prox_power=2.8, internal_central_power=2.4, center_boost=0.20,
                    FUNNEL_INFLUENCE_RANGE=35.0, FUNNEL_POWER=1.3, BASE_BOOST_WEIGHT=0.15,
                    band_width_m=180.0, blur_window_m=60.0, final_blur_m=12.0,
                    ANGLE_WEIGHT=0.50, ANGLE_POWER=1.4, BASE_ANGLE_WEIGHT=0.40):

    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)

    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    BASE = xp * (0.8 + 0.2 * yc)
    BASE = (BASE - BASE.min()) / (BASE.max() - BASE.min() + 1e-12)

    cy = FIELD_Y / 2.0
    fv = [(FIELD_X, cy-goal_width/2), (FIELD_X-penalty_depth, cy-penalty_width/2),
          (FIELD_X-penalty_depth, cy+penalty_width/2), (FIELD_X, cy+goal_width/2)]
    bpts = []
    for i in range(len(fv)):
        a, b = fv[i], fv[(i+1) % len(fv)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = max(2, int(round(math.hypot(dx, dy) / 0.5)))
        for t in np.linspace(0, 1, n, endpoint=False):
            bpts.append((a[0] + dx * t, a[1] + dy * t))
    bpts = np.array(bpts)

    fX = Xc.ravel()
    fY = Yc.ravel()
    md2 = np.full(fX.size, np.inf)
    for bp in bpts:
        dx = fX - bp[0]
        dy = fY - bp[1]
        np.minimum(md2, dx*dx + dy*dy, out=md2)
    adist = np.sqrt(md2).reshape(Xc.shape)

    infl = np.clip((1 - np.clip(adist / FUNNEL_INFLUENCE_RANGE, 0, 1))**FUNNEL_POWER, 0, 1)

    D = np.hypot(FIELD_X - Xc, cy - Yc)
    prox = 1 - np.clip(D / np.hypot(FIELD_X, FIELD_Y/2), 0, 1)
    cent = 1 - np.clip(np.abs((Yc - cy) / cy), 0, 1)
    ub = np.clip((prox_w * np.clip(prox**internal_prox_power, 0, 1) +
                  central_w * np.clip(cent**internal_central_power, 0, 1)) *
                 (1 + center_boost * prox), 0, 1)

    v1x = FIELD_X - Xc
    v1y = (cy + goal_width/2) - Yc
    v2x = FIELD_X - Xc
    v2y = (cy - goal_width/2) - Yc
    ca = np.clip((v1x*v2x + v1y*v2y) / (np.hypot(v1x,v1y) * np.hypot(v2x,v2y) + 1e-12), -1, 1)
    ang = np.arccos(ca)
    af = np.clip((ang / (ang.max() + 1e-12))**ANGLE_POWER, 0, 1)

    ub = np.clip(ub * ((1-ANGLE_WEIGHT) + ANGLE_WEIGHT*af), 0, 1)
    Bc = BASE * ((1-BASE_ANGLE_WEIGHT) + BASE_ANGLE_WEIGHT*af)
    Bc = (Bc - Bc.min()) / (Bc.max() - Bc.min() + 1e-12)
    XTB = Bc + infl * BASE_BOOST_WEIGHT * ub

    pw = FIELD_X / ncols_hr
    ph = FIELD_Y / nrows_hr
    rx = max(1, int(round((blur_window_m/pw)/2)))
    ry = max(1, int(round((blur_window_m/ph)/2)))

    def blur(a, rx, ry):
        H, W = a.shape
        p = np.pad(a, ((ry, ry), (rx, rx)), mode='edge').astype(np.float64)
        ii = p.cumsum(0).cumsum(1)
        s = ii[2*ry:2*ry+H, 2*rx:2*rx+W].copy()
        s += ii[:H, :W]
        s -= ii[:H, 2*rx:2*rx+W]
        s -= ii[2*ry:2*ry+H, :W]
        return s / ((2*ry+1)*(2*rx+1))

    w = 0.5 * (1 - np.cos(np.pi * np.clip(adist / band_width_m, 0, 1)))
    XTbl = w * XTB + (1-w) * blur(XTB, rx, ry)

    rf = max(1, int(round((final_blur_m/pw)/2)))
    rfy = max(1, int(round((final_blur_m/ph)/2)))
    XT = 0.85 * XTbl + 0.15 * blur(XTbl, rf, rfy)
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)

    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy*sub:(iy+1)*sub, ix*sub:(ix+1)*sub].mean()

    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX, 0, NX - 1))
    iy = int(np.clip((y / FIELD_Y) * NY, 0, NY - 1))
    return float(XT_GRID[iy, ix])

# =============================================================================
# Fixed data
# =============================================================================
def build_raw_passes():
    completed = [
        ("Seta 1", 26.75, 68.34, 8.97, 51.05), ("Seta 2", 31.24, 51.22, 34.57, 72.50),
        ("Seta 3", 36.06, 46.90, 44.37, 57.04), ("Seta 4", 48.36, 64.02, 58.17, 51.72),
        ("Seta 5", 58.17, 64.02, 62.49, 55.21), ("Seta 6", 54.51, 49.72, 64.82, 61.69),
        ("Seta 7", 42.21, 70.84, 34.90, 76.49), ("Seta 8", 43.54, 75.32, 36.73, 67.84),
        ("Seta 1", 32.24, 53.96, 6.81, 38.50), ("Seta 2", 33.57, 65.77, 36.56, 75.57),
        ("Seta 3", 37.39, 61.11, 43.04, 75.41), ("Seta 4", 65.49, 53.63, 56.18, 70.42),
        ("Seta 5", 55.68, 48.15, 46.87, 30.86), ("Seta 6", 52.02, 22.05, 46.70, 41.99),
        ("Seta 7", 62.16, 35.51, 71.80, 35.18), ("Seta 8", 54.02, 33.35, 63.99, 22.55),
        ("Seta 9", 60.00, 22.21, 76.62, 32.85), ("Seta 10", 87.10, 9.41, 77.45, 16.23),
        ("Seta 11", 62.66, 20.05, 117.18, 8.25), ("Seta 12", 98.90, 43.49, 103.22, 47.15),
        ("Seta 13", 70.31, 45.98, 82.28, 60.11), ("Seta 14", 85.10, 75.24, 101.39, 74.08),
        ("Seta 15", 53.18, 67.59, 39.05, 59.62), ("Seta 16", 55.18, 49.64, 54.85, 13.07),
        ("Seta 17", 68.64, 19.22, 49.03, 24.37), ("Seta 1", 53.35, 22.71, 59.34, 30.19),
        ("Seta 2", 44.37, 24.71, 40.05, 46.82), ("Seta 3", 43.88, 39.34, 41.38, 73.08),
        ("Seta 4", 56.84, 53.46, 70.81, 76.24), ("Seta 5", 82.77, 12.24, 91.42, 4.59),
        ("Seta 1", 108.04, 11.74, 115.69, 58.29), ("Seta 2", 93.08, 3.93, 111.03, 13.74),
        ("Seta 3", 84.60, 17.89, 96.74, 22.05), ("Seta 4", 58.34, 16.06, 65.65, 2.43),
        ("Seta 5", 52.02, 8.58, 44.37, 15.73), ("Seta 6", 61.00, 23.21, 49.36, 15.23),
        ("Seta 7", 32.74, 30.69, 50.03, 33.02), ("Seta 8", 51.85, 33.68, 60.66, 40.00),
        ("Seta 10", 79.95, 60.45, 98.23, 60.28), ("Seta 11", 31.24, 52.14, 39.05, 72.08),
        ("Seta 12", 39.72, 48.98, 33.40, 57.62), ("Seta 1", 70.64, 51.47, 61.00, 51.64),
        ("Seta 1", 21.27, 14.23, 29.25, 31.02), ("Seta 2", 29.41, 23.38, 34.40, 64.60),
        ("Seta 3", 41.55, 39.67, 41.88, 6.92), ("Seta 4", 44.54, 32.52, 43.54, 14.23),
        ("Seta 5", 23.59, 56.46, 34.57, 47.48), ("Seta 6", 30.58, 64.44, 21.10, 49.48),
        ("Seta 7", 33.07, 56.79, 49.53, 69.59), ("Seta 8", 33.24, 59.78, 44.04, 71.75),
        ("Seta 9", 61.50, 71.58, 54.68, 75.57), ("Seta 10", 63.16, 50.81, 78.45, 67.26),
        ("Seta 11", 63.49, 76.90, 84.44, 62.77), ("Seta 12", 76.96, 56.96, 86.93, 57.79),
        ("Seta 13", 82.61, 59.12, 96.41, 68.43), ("Seta 14", 79.78, 35.35, 106.21, 11.74),
        ("Seta 15", 45.37, 49.64, 40.72, 32.02),
        ("Seta 1", 28.08, 28.53, 29.75, 8.25), ("Seta 2", 33.74, 26.54, 29.41, 43.82),
        ("Seta 3", 28.08, 47.15, 31.57, 64.60), ("Seta 4", 39.39, 43.82, 51.69, 53.46),
        ("Seta 5", 43.88, 46.15, 55.84, 40.66), ("Seta 6", 47.03, 49.97, 44.04, 28.03),
        ("Seta 7", 47.53, 50.81, 71.97, 33.18), ("Seta 8", 67.65, 52.63, 64.32, 33.85),
        ("Seta 9", 73.63, 65.10, 69.31, 73.25), ("Seta 10", 77.29, 63.27, 79.12, 72.91),
        ("Seta 12", 81.61, 56.62, 93.91, 73.75), ("Seta 13", 86.43, 66.43, 81.78, 54.96),
        ("Seta 14", 111.03, 71.42, 99.56, 67.59), ("Seta 15", 89.76, 59.62, 97.74, 48.98),
        ("Seta 16", 88.43, 52.47, 96.41, 74.24), ("Seta 17", 87.93, 50.97, 77.12, 27.70),
        ("Seta 18", 81.61, 53.63, 74.30, 27.03), ("Seta 19", 79.28, 51.14, 94.91, 70.42),
        ("Seta 20", 52.85, 32.85, 65.49, 25.37), ("Seta 21", 82.77, 33.18, 69.31, 47.65),
        ("Seta 1", 39.39, 19.39, 52.35, 4.76), ("Seta 2", 63.82, 7.92, 72.63, 1.43),
        ("Seta 3", 70.47, 11.91, 80.95, 13.74), ("Seta 4", 64.49, 22.55, 97.24, 10.24),
        ("Seta 5", 32.07, 35.51, 43.04, 28.20), ("Seta 6", 53.52, 46.32, 54.02, 33.68),
        ("Seta 7", 77.12, 48.64, 84.94, 50.14), ("Seta 8", 78.12, 52.47, 117.52, 69.42),
        ("Seta 9", 88.76, 65.93, 97.40, 76.74), ("Seta 10", 82.61, 69.26, 86.60, 77.40),
        ("Seta 11", 78.62, 66.26, 79.62, 78.40), ("Seta 12", 83.61, 75.91, 62.49, 57.12),
        ("Seta 13", 34.40, 50.14, 88.76, 75.41), ("Seta 14", 56.68, 64.27, 78.29, 64.27),
        ("Seta 15", 51.85, 73.25, 54.18, 78.07), ("Seta 16", 41.05, 57.45, 46.04, 74.91),
        ("Seta 17", 37.39, 60.61, 41.71, 73.91), ("Seta 18", 30.41, 63.44, 36.89, 77.40),
        ("Seta 19", 26.09, 63.94, 28.42, 76.74), ("Seta 20", 22.43, 56.62, 22.10, 76.41),
        ("Seta 21", 33.90, 64.77, 25.42, 73.58),
    ]
    failed = [
        ("Seta 1", 53.35, 19.55, 73.96, 11.24), ("Seta 2", 63.82, 20.55, 88.76, 22.55),
        ("Seta 3", 85.60, 27.86, 94.41, 37.17), ("Seta 4", 77.79, 27.53, 96.41, 25.37),
        ("Seta 5", 91.09, 27.86, 109.54, 50.47), ("Seta 6", 58.17, 26.04, 95.41, 40.33),
        ("Seta 7", 53.35, 28.53, 73.80, 27.86), ("Seta 8", 53.35, 34.02, 84.60, 58.62),
        ("Seta 9", 56.18, 49.48, 97.07, 62.11), ("Seta 10", 34.23, 74.91, 65.65, 78.57),
        ("Seta 1", 78.62, 64.94, 96.57, 67.10), ("Seta 2", 85.43, 68.76, 106.05, 77.74),
        ("Seta 1", 72.14, 16.56, 78.45, 1.60), ("Seta 2", 79.62, 27.53, 97.07, 47.98),
        ("Seta 3", 91.75, 50.14, 109.70, 65.77), ("Seta 4", 96.41, 56.79, 107.04, 67.26),
        ("Seta 1", 41.88, 42.49, 56.18, 52.97), ("Seta 2", 37.56, 41.16, 46.37, 53.96),
        ("Seta 3", 54.68, 56.96, 54.85, 64.44), ("Seta 4", 51.69, 68.43, 66.15, 76.57),
    ]

    rows, idx = [], 1
    for label, x0, y0, x1, y1 in completed:
        rows.append({"number": idx, "seta": label, "x_start": x0, "y_start": y0, "x_end": x1, "y_end": y1, "is_won": True, "type": "PASS WON"})
        idx += 1
    for label, x0, y0, x1, y1 in failed:
        rows.append({"number": idx, "seta": label, "x_start": x0, "y_start": y0, "x_end": x1, "y_end": y1, "is_won": False, "type": "PASS LOST"})
        idx += 1
    return pd.DataFrame(rows)

def assign_pressure_coherent(df_input: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df_input.copy()
    tags = []
    for _, r in df.iterrows():
        x0, x1 = float(r["x_start"]), float(r["x_end"])
        p00, p10, p01, p11 = 0.62, 0.19, 0.13, 0.06
        if x0 >= FINAL_THIRD_LINE_X:
            p10 += 0.06; p11 += 0.04; p00 -= 0.07; p01 -= 0.03
        if x1 >= FINAL_THIRD_LINE_X:
            p01 += 0.06; p11 += 0.03; p00 -= 0.07; p10 -= 0.02
        probs = np.array([p00, p10, p01, p11], dtype=float)
        probs = np.clip(probs, 0.01, None)
        probs /= probs.sum()
        tags.append(rng.choice(["0-0", "1-0", "0-1", "1-1"], p=probs))
    df["pressure_tag"] = tags
    df["pressure_label"] = df["pressure_tag"].map(PRESSURE_LABEL)
    return df

@st.cache_data(show_spinner=False)
def build_dataset(seed=42):
    d = assign_pressure_coherent(build_raw_passes(), seed=seed).copy()
    d["outcome"] = np.where(d["is_won"], "completed", "incomplete")
    d["xt_start"] = d.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    d["xt_end"] = d.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
    d["ΔxT"] = np.where(d["is_won"], d["xt_end"] - d["xt_start"], 0.0)
    d["pressure_bonus"] = d["pressure_tag"].map(PRESSURE_BONUS).fillna(0.0)
    d["ΔxT_final"] = np.where(d["is_won"], d["ΔxT"] * (1 + d["pressure_bonus"]), 0.0)
    d["pass_distance"] = np.sqrt((d.x_end - d.x_start)**2 + (d.y_end - d.y_start)**2)

    # Global rank by ΔxT (completed only)
    d["rank_ΔxT"] = np.nan
    comp_idx = d[d["is_won"]].sort_values("ΔxT", ascending=False).index
    d.loc[comp_idx, "rank_ΔxT"] = np.arange(1, len(comp_idx) + 1, dtype=float)
    return d

# =============================================================================
# Drawing
# =============================================================================
def _base_pitch():
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e",
                  line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", lw=1.0, alpha=0.18)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.6, alpha=0.10, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig):
    fig.patches.append(FancyArrowPatch(
        (0.45,0.05),(0.55,0.05), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=15, linewidth=2, color="#cccccc"))
    fig.text(0.5,0.02,"Attacking Direction",ha="center",va="center",
             fontsize=9,color="#cccccc")

def _save_fig(fig):
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df: pd.DataFrame, mode: str):
    """
    mode:
      - "Top 10 by ΔxT"
      - "Errors only"
      - "All passes"
    """
    fig, ax, pitch = _base_pitch()

    if mode == "Top 10 by ΔxT":
        top10 = df[df["is_won"]].sort_values("ΔxT", ascending=False).head(10).copy()
        if top10.empty:
            ax.set_title("Pass Map — Top 10 by ΔxT", fontsize=12, color="#ffffff", pad=8)
            _attack_arrow(fig)
            return _save_fig(fig), ax, fig

        vmin = float(top10["ΔxT"].min())
        vmax = float(top10["ΔxT"].max())
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
        cmap = plt.cm.YlOrRd

        for _, row in top10.iterrows():
            color = cmap(norm(float(row["ΔxT"])))
            pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                         color=color,width=2.05,headwidth=2.35,headlength=2.35,
                         ax=ax,zorder=8,alpha=0.90)
            pitch.scatter(row.x_start,row.y_start,s=58,marker="o",color=color,
                          edgecolors="white",linewidths=0.9,ax=ax,zorder=9,alpha=0.92)

        ax.set_title("Pass Map — Top 10 by ΔxT", fontsize=12, color="#ffffff", pad=8)
        leg = ax.legend(handles=[
            Line2D([0],[0],color=plt.cm.YlOrRd(0.25),lw=2.8,label="Lower in Top 10",alpha=0.95),
            Line2D([0],[0],color=plt.cm.YlOrRd(0.95),lw=2.8,label="Higher in Top 10",alpha=0.95),
        ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
           facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
           labelspacing=0.5, borderpad=0.5)
        for t in leg.get_texts():
            t.set_color("white")
        leg.get_frame().set_alpha(0.92)

    elif mode == "Errors only":
        err = df[~df["is_won"]].copy()
        for _, row in err.iterrows():
            pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                         color=COLOR_FAIL_LIGHT,width=1.7,headwidth=2.2,headlength=2.2,
                         ax=ax,zorder=4,alpha=0.90)
            pitch.scatter(row.x_start,row.y_start,s=42,marker="o",color=COLOR_FAIL_LIGHT,
                          edgecolors="white",linewidths=0.7,ax=ax,zorder=5,alpha=0.92)
        ax.set_title("Pass Map — Errors only", fontsize=12, color="#ffffff", pad=8)

    else:  # All passes
        for _, row in df.iterrows():
            if bool(row["is_won"]):
                color, alpha = COLOR_ALL_PASS_LIGHT, 0.30
            else:
                color, alpha = COLOR_FAIL_LIGHT, 0.86

            pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                         color=color,width=1.45,headwidth=2.15,headlength=2.15,
                         ax=ax,zorder=4,alpha=alpha)
            pitch.scatter(row.x_start,row.y_start,s=38,marker="o",color=color,
                          edgecolors="white",linewidths=0.6,ax=ax,zorder=5,alpha=alpha)

        ax.set_title("Pass Map — All passes", fontsize=12, color="#ffffff", pad=8)
        leg = ax.legend(handles=[
            Line2D([0],[0],color=COLOR_ALL_PASS_LIGHT,lw=2.5,label="Completed",alpha=0.70),
            Line2D([0],[0],color=COLOR_FAIL_LIGHT,lw=2.5,label="Incomplete",alpha=0.90),
        ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
           facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
           labelspacing=0.5, borderpad=0.5)
        for t in leg.get_texts():
            t.set_color("white")
        leg.get_frame().set_alpha(0.92)

    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

def draw_corridor_heatmap(df: pd.DataFrame, title: str = "Zone Heatmap — Completed Passes"):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {
        "left":   (LANE_LEFT_MIN,  FIELD_Y),
        "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN),
        "right":  (0.0,            LANE_RIGHT_MAX),
    }
    counts = {}
    for cname,(y0,y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_,x1_ = x_bins[i],x_bins[i+1]
            mask = ((df_s["x_end"]>=x0_)&(df_s["x_end"]<x1_)
                    &(df_s["y_end"]>=y0)&(df_s["y_end"]<y1))
            arr[i] = int(mask.sum())
        counts[cname] = arr

    all_vals = np.concatenate([counts[c] for c in counts]) if len(df_s) else np.array([0])
    vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr",["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax*0.35)

    pitch = Pitch(pitch_type="statsbomb",pitch_color="#1a1a2e",line_color="#ffffff",line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W,FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)

    for cname,(y0,y1) in corridors.items():
        for i in range(6):
            x0_,x1_ = x_bins[i],x_bins[i+1]
            value = counts[cname][i]
            ax.add_patch(Rectangle((x0_,y0),x1_-x0_,y1-y0,
                                   facecolor=cmap(norm(value)),
                                   edgecolor=(1,1,1,0.12),lw=0.6,alpha=0.95,zorder=2))
            ax.text((x0_+x1_)/2,(y0+y1)/2,str(value),ha="center",va="center",
                    color="#000000" if value<=threshold else "#ffffff",
                    fontsize=11,fontweight="700" if value>=vmax*0.5 else "600",zorder=4)

    ax.set_title(title,fontsize=12,color="#ffffff",pad=8)
    ax.axhline(y=LANE_LEFT_MIN,color="#ffffff",lw=0.5,alpha=0.15,linestyle="--",zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX,color="#ffffff",lw=0.5,alpha=0.15,linestyle="--",zorder=3)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

def _top_zone_transitions(df_s: pd.DataFrame, top_k: int = 3):
    x_bins = np.linspace(0.0,FIELD_X,7)
    y_bins = np.array([0.0,LANE_RIGHT_MAX,LANE_LEFT_MIN,FIELD_Y])
    if df_s.empty:
        return [], x_bins, y_bins
    sx = np.clip(np.searchsorted(x_bins,df_s["x_start"].to_numpy(),side="right")-1,0,5)
    sy = np.clip(np.searchsorted(y_bins,df_s["y_start"].to_numpy(),side="right")-1,0,2)
    ex = np.clip(np.searchsorted(x_bins,df_s["x_end"].to_numpy(),  side="right")-1,0,5)
    ey = np.clip(np.searchsorted(y_bins,df_s["y_end"].to_numpy(),  side="right")-1,0,2)
    transitions = defaultdict(int)
    for a,b,c,d in zip(sx,sy,ex,ey):
        if int(a)==int(c) and int(b)==int(d):
            continue
        transitions[(int(a),int(b),int(c),int(d))] += 1
    links = sorted(transitions.items(),key=lambda kv:kv[1],reverse=True)[:top_k]
    return links, x_bins, y_bins

def draw_top_connection_minimaps(df: pd.DataFrame, top_k: int = 3,
                                 title: str = "Top Zone Connections — Completed Passes"):
    df_s = df[df["is_won"]].copy()
    links, x_bins, y_bins = _top_zone_transitions(df_s, top_k=top_k)
    x_cent = (x_bins[:-1]+x_bins[1:])/2.0
    y_cent = (y_bins[:-1]+y_bins[1:])/2.0
    max_cnt = max([v for _,v in links],default=1) if links else 1

    fig, axes = plt.subplots(1,top_k,figsize=(FIG_W*1.65,FIG_H*0.82),dpi=FIG_DPI)
    if top_k == 1:
        axes = [axes]
    fig.set_facecolor("#1a1a2e")
    pitch = Pitch(pitch_type="statsbomb",pitch_color="#1a1a2e",
                  line_color="#ffffff",line_alpha=0.90)

    for idx, ax in enumerate(axes):
        pitch.draw(ax=ax)
        ax.axhline(y=LANE_LEFT_MIN,color="#ffffff",lw=0.4,alpha=0.12,linestyle="--")
        ax.axhline(y=LANE_RIGHT_MAX,color="#ffffff",lw=0.4,alpha=0.12,linestyle="--")

        if idx >= len(links):
            ax.set_title("—",fontsize=9,color="#dbeafe",pad=4)
            continue

        (ix0,iy0,ix1,iy1),cnt = links[idx]
        x0,y0 = float(x_cent[ix0]),float(y_cent[iy0])
        x1,y1 = float(x_cent[ix1]),float(y_cent[iy1])
        rel = cnt/max_cnt
        color = plt.cm.Blues(0.40+0.55*rel)

        ax.add_patch(Rectangle((x_bins[ix0],y_bins[iy0]),
                               x_bins[ix0+1]-x_bins[ix0],y_bins[iy0+1]-y_bins[iy0],
                               facecolor=(0.20,0.45,0.95,0.18),edgecolor=(1,1,1,0.18),lw=0.6,zorder=2))
        ax.add_patch(Rectangle((x_bins[ix1],y_bins[iy1]),
                               x_bins[ix1+1]-x_bins[ix1],y_bins[iy1+1]-y_bins[iy1],
                               facecolor=(0.02,0.70,0.55,0.18),edgecolor=(1,1,1,0.18),lw=0.6,zorder=2))

        if ix0==ix1 and iy0==iy1:
            ax.scatter([x0],[y0],s=40+80*rel,c=[color],marker="o",
                       edgecolors="white",linewidths=0.5,alpha=0.35+0.60*rel,zorder=5)
        else:
            rad = float(np.clip(0.10*np.sign((ix1-ix0)+0.4*(iy1-iy0)),-0.30,0.30))
            ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>",mutation_scale=10+9*rel,
                                         lw=1.2+4.2*rel,color=color,alpha=0.35+0.60*rel,zorder=4))
        ax.text((x0+x1)/2,(y0+y1)/2,f"{cnt}",color="#e5efff",fontsize=9,ha="center",va="center",zorder=7,
                bbox=dict(boxstyle="round,pad=0.18",fc=(0.06,0.09,0.14,0.80),ec="none"))
        ax.set_title(f"#{idx+1}  ·  {cnt}×",fontsize=9,color="#dbeafe",pad=4)

    fig.suptitle(title,fontsize=11,color="#ffffff",y=0.99)
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf,format="png",dpi=FIG_DPI,facecolor=fig.get_facecolor(),bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf), axes, fig

# =============================================================================
# Stats
# =============================================================================
def compute_stats(df_in: pd.DataFrame) -> dict:
    total = len(df_in)
    completed = int(df_in["is_won"].sum())
    incomplete = total - completed
    accuracy = round(completed / max(total, 1) * 100, 2)

    comp = df_in[df_in["is_won"]]
    sum_dx = float(comp["ΔxT"].sum()) if not comp.empty else 0.0
    mean_dx = float(comp["ΔxT"].mean()) if not comp.empty else 0.0
    cnt_ge = int((comp["ΔxT"] >= DELTA_THR).sum()) if not comp.empty else 0
    pct_ge = round(cnt_ge / max(total, 1) * 100, 2)

    by_pressure = (
        df_in.groupby("pressure_tag", dropna=False)
        .agg(
            total=("number","count"),
            completed=("is_won","sum"),
            incomplete=("is_won", lambda s: int((~s).sum())),
            sum_ΔxT=("ΔxT","sum"),
            mean_ΔxT=("ΔxT","mean"),
            ge_thr=("ΔxT", lambda s: int((s >= DELTA_THR).sum())),
        )
        .reset_index()
    )
    by_pressure["accuracy_pct"] = np.where(by_pressure["total"] > 0, by_pressure["completed"] / by_pressure["total"] * 100, 0)
    by_pressure["ge_thr_pct_total"] = np.where(by_pressure["total"] > 0, by_pressure["ge_thr"] / by_pressure["total"] * 100, 0)
    by_pressure["pressure_label"] = by_pressure["pressure_tag"].map(PRESSURE_LABEL)
    by_pressure = by_pressure.sort_values("pressure_tag")

    return {
        "total": total,
        "completed": completed,
        "incomplete": incomplete,
        "accuracy": accuracy,
        "sum_ΔxT": sum_dx,
        "mean_ΔxT": mean_dx,
        "cnt_ge": cnt_ge,
        "pct_ge": pct_ge,
        "by_pressure": by_pressure
    }

# =============================================================================
# Data
# =============================================================================
with st.sidebar:
    st.markdown("### Settings")
    seed = st.number_input("Pressure seed", min_value=1, max_value=999999, value=42, step=1)

df = build_dataset(seed=int(seed))

# =============================================================================
# Tabs
# =============================================================================
tab_map, tab_analysis = st.tabs(["📋 Map", "📈 Analysis"])

# =============================================================================
# TAB: MAP
# =============================================================================
with tab_map:
    st.caption("Click the origin marker on the pass map to inspect an event.")
    col_filters, col_field = st.columns([0.9, 2], gap="large")

    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown("### 🧭 View mode")
        view_mode = st.radio(
            "Map view",
            ["Top 10 by ΔxT", "Errors only", "All passes"],
            index=0
        )
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown("### ⚡ Pressure")
        pressure_filter = st.radio(
            "Pressure filter",
            ["All", "0-0", "1-0", "0-1", "1-1"],
            index=0
        )
        st.markdown('</div>', unsafe_allow_html=True)

    df_base = df.copy()
    if pressure_filter != "All":
        df_base = df_base[df_base["pressure_tag"] == pressure_filter].reset_index(drop=True)

    with col_field:
        img_obj, ax, fig = draw_pass_map(df_base, view_mode)
        click = streamlit_image_coordinates(img_obj, width=780, key="main_map")

        selected_pass = None
        if click is not None and len(df_base) > 0:
            rw,rh = img_obj.size
            px = click["x"]*(rw/click["width"])
            py = click["y"]*(rh/click["height"])
            fx,fy = ax.transData.inverted().transform((px,rh-py))
            df_sel = df_base.copy()
            df_sel["_dist"] = np.sqrt((df_sel.x_start-fx)**2+(df_sel.y_start-fy)**2)
            cands = df_sel[df_sel["_dist"]<5.0].sort_values("_dist")
            if not cands.empty:
                selected_pass = cands.iloc[0]
        plt.close(fig)

        st.divider()
        st.subheader("Selected Event")
        if selected_pass is None:
            st.info("Click an origin marker on the map to inspect an event.")
        else:
            rank_txt = "—"
            if pd.notna(selected_pass["rank_ΔxT"]):
                rank_txt = f"#{int(selected_pass['rank_ΔxT'])}"

            status = "✅ Completed" if selected_pass["is_won"] else "❌ Incomplete"
            st.success(
                f"Pass #{int(selected_pass['number'])} — {selected_pass['type']} | "
                f"{status} | ΔxT rank: {rank_txt}"
            )

            c1,c2 = st.columns(2)
            c1.write(f"**Origin:** ({selected_pass.x_start:.2f}, {selected_pass.y_start:.2f})")
            c2.write(f"**Destination:** ({selected_pass.x_end:.2f}, {selected_pass.y_end:.2f})")

            c3,c4,c5 = st.columns(3)
            with c3: st.metric("xT start", f"{selected_pass.xt_start:.4f}")
            with c4: st.metric("xT end", f"{selected_pass.xt_end:.4f}")
            with c5: st.metric("ΔxT", f"{selected_pass['ΔxT']:.4f}")

            p1,p2 = st.columns(2)
            p1.write(f"**Pressure:** {selected_pass['pressure_tag']} ({selected_pass['pressure_label']})")
            p2.write(f"**Pressure bonus:** {selected_pass['pressure_bonus']:.2f}")

        with st.expander("📊 Full table"):
            cols = ["number","seta","type","pressure_tag","pressure_label","outcome",
                    "x_start","y_start","x_end","y_end","xt_start","xt_end","ΔxT","rank_ΔxT","pass_distance"]
            st.dataframe(
                df_base[cols].style.format({
                    "x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}","y_end":"{:.2f}",
                    "xt_start":"{:.4f}","xt_end":"{:.4f}","ΔxT":"{:.4f}",
                    "rank_ΔxT":"{:.0f}","pass_distance":"{:.1f}"
                }),
                use_container_width=True,
                height=420
            )

# =============================================================================
# TAB: ANALYSIS
# =============================================================================
with tab_analysis:
    st.caption("Heatmap and zone interactions are available in this tab.")
    col_left, col_right = st.columns([2,1], gap="large")

    with col_left:
        # State reset for zone interaction
        for key, default in [
            ("heat_sel_analysis", None),
            ("last_pressure_analysis", pressure_filter),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default
        if st.session_state["last_pressure_analysis"] != pressure_filter:
            st.session_state["heat_sel_analysis"] = None
            st.session_state["last_pressure_analysis"] = pressure_filter

        # Analysis df respects pressure filter
        df_an = df.copy()
        if pressure_filter != "All":
            df_an = df_an[df_an["pressure_tag"] == pressure_filter].reset_index(drop=True)

        st.markdown('<h4 style="color:#ffffff;margin:6px 0 6px 0;">Zone Heatmap</h4>', unsafe_allow_html=True)
        heat_img, hax, hfig = draw_corridor_heatmap(df_an)
        heat_click = streamlit_image_coordinates(heat_img, width=780, key="analysis_heat")
        if heat_click is not None:
            rw,rh = heat_img.size
            px = heat_click["x"]*(rw/heat_click["width"])
            py = heat_click["y"]*(rh/heat_click["height"])
            fx,fy = hax.transData.inverted().transform((px,rh-py))

            xb = np.linspace(0,FIELD_X,7)
            ix = max(0,min(5,np.searchsorted(xb,fx,side="right")-1))
            x0h,x1h = xb[ix],xb[ix+1]

            if fy >= LANE_LEFT_MIN:
                cn,y0h,y1h = "left", LANE_LEFT_MIN, FIELD_Y
            elif fy < LANE_RIGHT_MAX:
                cn,y0h,y1h = "right", 0.0, LANE_RIGHT_MAX
            else:
                cn,y0h,y1h = "center", LANE_RIGHT_MAX, LANE_LEFT_MIN

            st.session_state["heat_sel_analysis"] = {
                "ix":int(ix),"corridor":cn,
                "x0":float(x0h),"x1":float(x1h),
                "y0":float(y0h),"y1":float(y1h)
            }
        plt.close(hfig)

        st.markdown('<h4 style="color:#ffffff;margin:14px 0 4px 0;">Top Zone Connections</h4>', unsafe_allow_html=True)
        mini_img, _, mini_fig = draw_top_connection_minimaps(df_an, top_k=3)
        st.image(mini_img, use_container_width=True)
        plt.close(mini_fig)

        st.markdown('<h4 style="color:#ffffff;margin:14px 0 6px 0;">Zone-filtered Map (All passes)</h4>', unsafe_allow_html=True)
        if st.button("Clear Zone Filter", key="analysis_clear_zone"):
            st.session_state["heat_sel_analysis"] = None

        df_zone = df_an.copy()
        if st.session_state["heat_sel_analysis"] is not None:
            sel = st.session_state["heat_sel_analysis"]
            df_zone = df_zone[
                (df_zone["x_end"]>=sel["x0"])&(df_zone["x_end"]<sel["x1"])
                &(df_zone["y_end"]>=sel["y0"])&(df_zone["y_end"]<sel["y1"])
            ].reset_index(drop=True)

        zone_map_img, _, zone_map_fig = draw_pass_map(df_zone, "All passes")
        st.image(zone_map_img, use_container_width=True)
        plt.close(zone_map_fig)

        if st.session_state["heat_sel_analysis"] is not None:
            sel = st.session_state["heat_sel_analysis"]
            n = int(((df_an["x_end"]>=sel["x0"])&(df_an["x_end"]<sel["x1"])
                     &(df_an["y_end"]>=sel["y0"])&(df_an["y_end"]<sel["y1"])).sum())
            st.markdown(
                f"<div style='color:#ffffff;margin-top:6px;'>"
                f"<strong>Zone filter active:</strong> channel <code>{sel['corridor']}</code>, "
                f"column #{sel['ix']+1} — {n} passes</div>",
                unsafe_allow_html=True
            )

    with col_right:
        s = compute_stats(df_an)

        with st.expander("📋 General Stats", expanded=True):
            st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
            r1,r2,r3 = st.columns(3)
            with r1: small_metric("Total passes", f"{s['total']}")
            with r2: small_metric("Completed", f"{s['completed']}")
            with r3: small_metric("Accuracy", f"{s['accuracy']:.1f}%")

            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
            x1,x2 = st.columns(2)
            with x1: small_metric("Σ ΔxT", f"{s['sum_ΔxT']:.3f}")
            with x2: small_metric("Mean ΔxT", f"{s['mean_ΔxT']:.3f}")

            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
            g1,g2 = st.columns(2)
            with g1: small_metric(f"ΔxT ≥ {DELTA_THR:.2f}", f"{s['cnt_ge']}")
            with g2: small_metric("Share of total", f"{s['pct_ge']:.1f}%")

        with st.expander("⚡ Pressure Stats", expanded=True):
            for _, row in s["by_pressure"].iterrows():
                st.markdown(
                    f"<div class='stats-section-title'>{row['pressure_tag']} — {row['pressure_label']}</div>",
                    unsafe_allow_html=True
                )
                a1,a2,a3 = st.columns(3)
                with a1: small_metric("Total", f"{int(row['total'])}")
                with a2: small_metric("Completed", f"{int(row['completed'])}")
                with a3: small_metric("Accuracy", f"{float(row['accuracy_pct']):.1f}%")

                b1,b2 = st.columns(2)
                with b1: small_metric("Σ ΔxT", f"{float(row['sum_ΔxT']):.3f}")
                with b2: small_metric("Mean ΔxT", f"{float(row['mean_ΔxT']):.3f}")

                c1,c2 = st.columns(2)
                with c1: small_metric(f"ΔxT ≥ {DELTA_THR:.2f}", f"{int(row['ge_thr'])}")
                with c2: small_metric("Share in pressure group", f"{float(row['ge_thr_pct_total']):.1f}%")
                st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)

        st.caption("Map modes: Top 10 by ΔxT, Errors only, or All passes.")
