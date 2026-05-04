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
st.set_page_config(layout="wide", page_title="Pass Map Dashboard — Pressão + deltaxT")

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

st.title("Pass Map Dashboard — Pressão + deltaxT")

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

# Visual map rules (pedido)
DELTA_XT_THRESHOLD = 0.05
COLOR_FAIL = "#E07070"      # errados
COLOR_PASS = "#2F80ED"      # válidos no mapa (>=0.05)
COLOR_TOP10 = "#D4AF37"     # top10
ALPHA_FAIL = 0.72
ALPHA_PASS = 0.58
ALPHA_TOP10 = 0.90

PRESSURE_BONUS = {
    "0-0": 0.00,
    "1-0": 0.06,
    "0-1": 0.06,
    "1-1": 0.12,
}
PRESSURE_LABEL = {
    "0-0": "Sem pressão",
    "1-0": "Pressão na origem",
    "0-1": "Pressão no destino",
    "1-1": "Pressão em ambos",
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
        p = np.pad(a, ((ry, ry), (rx, rx)), mode="edge").astype(np.float64)
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
# Fixed data (1 partida consolidada)
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
    d["deltaxT"] = np.where(d["is_won"], d["xt_end"] - d["xt_start"], 0.0)  # sem pressão no nome
    d["pressure_bonus"] = d["pressure_tag"].map(PRESSURE_BONUS).fillna(0.0)
    d["deltaxT_final"] = np.where(d["is_won"], d["deltaxT"] * (1 + d["pressure_bonus"]), 0.0)
    d["pass_distance"] = np.sqrt((d.x_end - d.x_start)**2 + (d.y_end - d.y_start)**2)

    # rank global por deltaxT (somente completos)
    d["rank_deltaxT"] = np.nan
    comp_idx = d[d["is_won"]].sort_values("deltaxT", ascending=False).index
    d.loc[comp_idx, "rank_deltaxT"] = np.arange(1, len(comp_idx) + 1, dtype=float)
    return d

# =============================================================================
# Draw helpers
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

def draw_corridor_heatmap(df: pd.DataFrame,
                          title: str = "Zone Heatmap — Completed Passes"):
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
    cmap = LinearSegmentedColormap.from_list(
        "wr",["#ffffff","#ffecec","#ffbfbf","#ff8080","#ff3b3b","#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax)
    threshold = max(1, vmax*0.35)

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
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
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff",lw=0.5,alpha=0.15,linestyle="--",zorder=3)
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
        ax.axhline(y=LANE_LEFT_MIN, color="#ffffff",lw=0.4,alpha=0.12,linestyle="--")
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

def draw_pass_map(df: pd.DataFrame, title: str, top10_ids: set[int]):
    fig, ax, pitch = _base_pitch()

    base_df = df[~df["number"].isin(top10_ids)]
    top_df = df[df["number"].isin(top10_ids)]

    # base
    for _, row in base_df.iterrows():
        if not bool(row["is_won"]):
            color, alpha, lw, sz = COLOR_FAIL, ALPHA_FAIL, 1.7, 44
        else:
            # só chegou aqui se deltaxT >= 0.05 (filtrado antes)
            color, alpha, lw, sz = COLOR_PASS, ALPHA_PASS, 1.65, 44

        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=color,width=lw,headwidth=2.25,headlength=2.25,ax=ax,zorder=4,alpha=alpha)
        pitch.scatter(row.x_start,row.y_start,s=sz,marker="o",color=color,
                      edgecolors="white",linewidths=0.75,ax=ax,zorder=5,alpha=alpha)

    # top10 por cima
    for _, row in top_df.iterrows():
        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=COLOR_TOP10,width=2.05,headwidth=2.35,headlength=2.35,ax=ax,zorder=8,alpha=ALPHA_TOP10)
        pitch.scatter(row.x_start,row.y_start,s=58,marker="o",color=COLOR_TOP10,
                      edgecolors="white",linewidths=0.9,ax=ax,zorder=9,alpha=ALPHA_TOP10)

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    leg = ax.legend(handles=[
        Line2D([0],[0],color=COLOR_PASS, lw=2.6,label=f"Completed (deltaxT ≥ {DELTA_XT_THRESHOLD:.2f})", alpha=0.90),
        Line2D([0],[0],color=COLOR_FAIL, lw=2.6,label="Incomplete", alpha=0.90),
        Line2D([0],[0],color=COLOR_TOP10,lw=2.8,label="Top 10 deltaxT", alpha=0.95),
    ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
       labelspacing=0.5, borderpad=0.5)
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.92)

    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

# =============================================================================
# Stats helpers
# =============================================================================
def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    completed = int(df["is_won"].sum())
    acc = round(completed / max(total, 1) * 100, 2)

    comp = df[df["is_won"]]
    sum_dxt = float(comp["deltaxT"].sum()) if not comp.empty else 0.0
    mean_dxt = float(comp["deltaxT"].mean()) if not comp.empty else 0.0
    pos_count = int((comp["deltaxT"] > 0).sum()) if not comp.empty else 0
    ge005_count = int((comp["deltaxT"] >= DELTA_XT_THRESHOLD).sum()) if not comp.empty else 0

    by_pressure = (
        df.groupby("pressure_tag", dropna=False)
          .agg(
              total=("number", "count"),
              completed=("is_won", "sum"),
              incomplete=("is_won", lambda s: int((~s).sum())),
              sum_deltaxT=("deltaxT", "sum"),
              mean_deltaxT=("deltaxT", "mean"),
              ge005=("deltaxT", lambda s: int((s >= DELTA_XT_THRESHOLD).sum())),
          )
          .reset_index()
    )
    by_pressure["accuracy_pct"] = np.where(
        by_pressure["total"] > 0,
        by_pressure["completed"] / by_pressure["total"] * 100,
        0.0
    )
    by_pressure["pressure_label"] = by_pressure["pressure_tag"].map(PRESSURE_LABEL)
    by_pressure = by_pressure.sort_values("pressure_tag")

    return {
        "total_passes": total,
        "completed_passes": completed,
        "incomplete_passes": total - completed,
        "accuracy_pct": acc,
        "sum_deltaxT": sum_dxt,
        "mean_deltaxT": mean_dxt,
        "positive_deltaxT_count": pos_count,
        "ge005_count": ge005_count,
        "by_pressure": by_pressure
    }

# =============================================================================
# App data
# =============================================================================
with st.sidebar:
    st.markdown("### Configuração")
    seed = st.number_input("Seed pressão", min_value=1, max_value=999999, value=42, step=1)

df = build_dataset(seed=int(seed))

# top10 global por deltaxT
top10_global = (
    df[df["is_won"]]
    .sort_values("deltaxT", ascending=False)
    .head(10)
)
top10_ids = set(top10_global["number"].astype(int).tolist())

# =============================================================================
# Tabs
# =============================================================================
tab_passmap, tab_analysis = st.tabs(["📋 Pass Map", "📈 Análise"])

# =============================================================================
# TAB 1
# =============================================================================
with tab_passmap:
    st.caption("Click the origin dot on the pass map to inspect an event.")
    col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

        st.markdown("### ⚡ Pressão")
        pressure_filter = st.radio(
            "Filter by pressure",
            ["All", "0-0", "1-0", "0-1", "1-1"],
            index=0,
            key="pm_pressure"
        )

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        st.markdown("### 🎯 Pass Filter")
        pass_filter = st.radio(
            "Filter passes",
            ["All Passes", "Completed Only", "Incomplete Only"],
            index=0,
            key="pm_filter"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # base filters
    df_base = df.copy()
    if pressure_filter != "All":
        df_base = df_base[df_base["pressure_tag"] == pressure_filter]

    if pass_filter == "Completed Only":
        df_base = df_base[df_base["is_won"]]
    elif pass_filter == "Incomplete Only":
        df_base = df_base[~df_base["is_won"]]

    df_base = df_base.reset_index(drop=True)

    # aplicar regra do mapa:
    # remove completos com deltaxT < 0.05; mantém errados; mantém top10
    df_for_map = df_base[
        (~df_base["is_won"]) |
        (df_base["deltaxT"] >= DELTA_XT_THRESHOLD) |
        (df_base["number"].isin(top10_ids))
    ].reset_index(drop=True)

    for key,default in [("heat_sel_pm",None),("last_pressure_pm",pressure_filter),("last_filter_pm",pass_filter)]:
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state["last_pressure_pm"] != pressure_filter:
        st.session_state["heat_sel_pm"] = None
        st.session_state["last_pressure_pm"] = pressure_filter
    if st.session_state["last_filter_pm"] != pass_filter:
        st.session_state["heat_sel_pm"] = None
        st.session_state["last_filter_pm"] = pass_filter

    with col_field:
        DW = 780
        pm_placeholder = st.empty()

        # Heatmap + interação de zona (igual dinâmica do app base)
        st.markdown('<h4 style="color:#ffffff;margin:6px 0 6px 0;">Zone Heatmap</h4>', unsafe_allow_html=True)
        heat_img,hax,hfig = draw_corridor_heatmap(df_base)
        heat_click = streamlit_image_coordinates(heat_img,width=DW,key="pm_heat")
        if heat_click is not None:
            rw,rh = heat_img.size
            px = heat_click["x"]*(rw/heat_click["width"])
            py = heat_click["y"]*(rh/heat_click["height"])
            fx,fy = hax.transData.inverted().transform((px,rh-py))
            xb = np.linspace(0,FIELD_X,7)
            ix = max(0,min(5,np.searchsorted(xb,fx,side="right")-1))
            x0h,x1h = xb[ix],xb[ix+1]
            if fy >= LANE_LEFT_MIN:      cn,y0h,y1h = "left",  LANE_LEFT_MIN, FIELD_Y
            elif fy < LANE_RIGHT_MAX:    cn,y0h,y1h = "right", 0.0,           LANE_RIGHT_MAX
            else:                        cn,y0h,y1h = "center",LANE_RIGHT_MAX,LANE_LEFT_MIN
            st.session_state["heat_sel_pm"] = {
                "ix":int(ix),"corridor":cn,
                "x0":float(x0h),"x1":float(x1h),
                "y0":float(y0h),"y1":float(y1h)}
        plt.close(hfig)

        st.markdown('<h4 style="color:#ffffff;margin:14px 0 4px 0;">Top Zone Connections</h4>', unsafe_allow_html=True)
        mini_img,_,mini_fig = draw_top_connection_minimaps(df_base,top_k=3)
        st.image(mini_img,use_container_width=True)
        plt.close(mini_fig)

        with pm_placeholder.container():
            st.markdown('<h4 style="color:#ffffff;margin:0 0 6px 0;">Pass Map</h4>', unsafe_allow_html=True)
            if st.button("Clear Zone Filter",key="pm_clear"):
                st.session_state["heat_sel_pm"] = None

            df_to_draw = df_for_map.copy()
            if st.session_state["heat_sel_pm"] is not None:
                sel = st.session_state["heat_sel_pm"]
                df_to_draw = df_to_draw[
                    (df_to_draw["x_end"]>=sel["x0"])&(df_to_draw["x_end"]<sel["x1"])
                    &(df_to_draw["y_end"]>=sel["y0"])&(df_to_draw["y_end"]<sel["y1"])
                ].reset_index(drop=True)

            img_obj,ax,fig = draw_pass_map(df_to_draw,title="Pass Map — Pressão + deltaxT", top10_ids=top10_ids)
            click = streamlit_image_coordinates(img_obj,width=DW,key="pm_map")

        selected_pass = None
        if click is not None and len(df_to_draw) > 0:
            rw,rh = img_obj.size
            px = click["x"]*(rw/click["width"])
            py = click["y"]*(rh/click["height"])
            fx,fy = ax.transData.inverted().transform((px,rh-py))
            df_sel = df_to_draw.copy()
            df_sel["_dist"] = np.sqrt((df_sel.x_start-fx)**2+(df_sel.y_start-fy)**2)
            cands = df_sel[df_sel["_dist"]<5.0].sort_values("_dist")
            if not cands.empty:
                selected_pass = cands.iloc[0]
        plt.close(fig)

        if st.session_state["heat_sel_pm"] is not None:
            sel = st.session_state["heat_sel_pm"]
            n = int(((df_for_map["x_end"]>=sel["x0"])&(df_for_map["x_end"]<sel["x1"])
                     &(df_for_map["y_end"]>=sel["y0"])&(df_for_map["y_end"]<sel["y1"])).sum())
            st.markdown(
                f"<div style='color:#ffffff;margin-top:6px;'>"
                f"<strong>Zone filter active:</strong> channel <code>{sel['corridor']}</code>, "
                f"column #{sel['ix']+1} — {n} passes</div>",
                unsafe_allow_html=True
            )

        st.divider()
        st.subheader("Selected Event")
        if selected_pass is None:
            st.info("Click an origin dot on the pass map to inspect an event.")
        else:
            status = "✅ Completed" if selected_pass["is_won"] else "❌ Incomplete"
            rank_txt = "-"
            if pd.notna(selected_pass["rank_deltaxT"]):
                rank_txt = f"#{int(selected_pass['rank_deltaxT'])}"

            top_tag = " · 🏅 Top 10" if int(selected_pass["number"]) in top10_ids else ""
            st.success(
                f"Pass #{int(selected_pass['number'])} — {selected_pass['type']} | "
                f"{status}{top_tag} | Rank deltaxT: {rank_txt}"
            )

            c1,c2 = st.columns(2)
            c1.write(f"**Origin:** ({selected_pass.x_start:.2f}, {selected_pass.y_start:.2f})")
            c2.write(f"**Destination:** ({selected_pass.x_end:.2f}, {selected_pass.y_end:.2f})")

            p1,p2 = st.columns(2)
            p1.write(f"**Pressure:** {selected_pass['pressure_tag']} ({selected_pass['pressure_label']})")
            p2.write(f"**Pressure Bonus:** {selected_pass['pressure_bonus']:.2f}")

            m1,m2,m3 = st.columns(3)
            with m1: st.metric("xT Start", f"{selected_pass.xt_start:.4f}")
            with m2: st.metric("xT End", f"{selected_pass.xt_end:.4f}")
            with m3: st.metric("deltaxT", f"{selected_pass.deltaxT:.4f}")

            st.metric("Pass Distance", f"{selected_pass.pass_distance:.1f} m")

        with st.expander("📊 Full Pass Data Table"):
            dc = ["number","seta","type","pressure_tag","pressure_label","outcome",
                  "x_start","y_start","x_end","y_end","xt_start","xt_end",
                  "deltaxT","deltaxT_final","rank_deltaxT","pass_distance"]
            st.dataframe(
                df_to_draw[dc].style.format(
                    {"x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}","y_end":"{:.2f}",
                     "xt_start":"{:.4f}","xt_end":"{:.4f}","deltaxT":"{:.4f}",
                     "deltaxT_final":"{:.4f}","rank_deltaxT":"{:.0f}","pass_distance":"{:.1f}"}
                ),
                use_container_width=True,height=420
            )

    with col_stats:
        s = compute_stats(df_base)

        with st.expander("📋 General Statistics",expanded=True):
            st.markdown('<div class="stats-section-title">Overview</div>',unsafe_allow_html=True)
            r1,r2,r3 = st.columns(3)
            with r1: small_metric("Total Passes", f"{s['total_passes']}")
            with r2: small_metric("Completed", f"{s['completed_passes']}")
            with r3: small_metric("Accuracy", f"{s['accuracy_pct']:.1f}%")

            st.markdown("<hr style='margin:6px 0 8px 0;'>",unsafe_allow_html=True)
            x1,x2,x3 = st.columns(3)
            with x1: small_metric("Σ deltaxT", f"{s['sum_deltaxT']:.3f}")
            with x2: small_metric("Mean deltaxT", f"{s['mean_deltaxT']:.3f}")
            with x3: small_metric(f"deltaxT ≥ {DELTA_XT_THRESHOLD:.2f}", f"{s['ge005_count']}")

            st.markdown("<hr style='margin:6px 0 8px 0;'>",unsafe_allow_html=True)
            small_metric("Positive deltaxT", f"{s['positive_deltaxT_count']}")

        with st.expander("⚡ Pressure + deltaxT Detail", expanded=True):
            byp = s["by_pressure"]
            for _, row in byp.iterrows():
                st.markdown(
                    f"<div class='stats-section-title'>{row['pressure_tag']} — {row['pressure_label']}</div>",
                    unsafe_allow_html=True
                )
                p1,p2,p3 = st.columns(3)
                with p1: small_metric("Total", f"{int(row['total'])}")
                with p2: small_metric("Completed", f"{int(row['completed'])}")
                with p3: small_metric("Accuracy", f"{float(row['accuracy_pct']):.1f}%")
                q1,q2,q3 = st.columns(3)
                with q1: small_metric("Σ deltaxT", f"{float(row['sum_deltaxT']):.3f}")
                with q2: small_metric("Mean deltaxT", f"{float(row['mean_deltaxT']):.3f}")
                with q3: small_metric(f"deltaxT ≥ {DELTA_XT_THRESHOLD:.2f}", f"{int(row['ge005'])}")
                st.markdown("<hr style='margin:6px 0 8px 0;'>",unsafe_allow_html=True)

        st.divider()
        st.caption("Map rules: deltaxT < 0.05 removed (except incomplete and Top 10). "
                   "Blue = completed, Red = incomplete, Gold = Top 10 deltaxT.")

# =============================================================================
# TAB 2 - Analysis
# =============================================================================
with tab_analysis:
    st.subheader("Top 20 by deltaxT")
    top20 = (
        df[df["is_won"]]
        .sort_values("deltaxT", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20["rank"] = np.arange(1, len(top20)+1)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Passes", f"{len(df)}")
    c2.metric("Completed", f"{int(df['is_won'].sum())}")
    c3.metric("Σ deltaxT (Top20)", f"{top20['deltaxT'].sum():.3f}")
    c4.metric("Mean deltaxT (Top20)", f"{top20['deltaxT'].mean() if len(top20)>0 else 0:.3f}")

    st.markdown('<h4 style="color:#ffffff;margin:8px 0 4px 0;">Map — Top 20 deltaxT</h4>', unsafe_allow_html=True)
    top20_img, top20_ax, top20_fig = draw_pass_map(top20, "Top 20 Passes by deltaxT", top10_ids=top10_ids)
    top20_click = streamlit_image_coordinates(top20_img, width=980, key="top20_map_click")

    top20_selected = None
    if top20_click is not None and len(top20) > 0:
        rw,rh = top20_img.size
        px = top20_click["x"]*(rw/top20_click["width"])
        py = top20_click["y"]*(rh/top20_click["height"])
        fx,fy = top20_ax.transData.inverted().transform((px,rh-py))
        df_sel = top20.copy()
        df_sel["_dist"] = np.sqrt((df_sel.x_start-fx)**2+(df_sel.y_start-fy)**2)
        cands = df_sel[df_sel["_dist"]<5.0].sort_values("_dist")
        if not cands.empty:
            top20_selected = cands.iloc[0]
    plt.close(top20_fig)

    if top20_selected is None:
        st.info("Click an origin dot on the Top 20 map to inspect an event.")
    else:
        st.success(
            f"Rank #{int(top20_selected['rank'])} — Pass #{int(top20_selected['number'])} "
            f"| deltaxT: {top20_selected['deltaxT']:.4f}"
        )
        a1,a2,a3 = st.columns(3)
        a1.write(f"**Origin:** ({top20_selected.x_start:.2f}, {top20_selected.y_start:.2f})")
        a2.write(f"**Destination:** ({top20_selected.x_end:.2f}, {top20_selected.y_end:.2f})")
        a3.write(f"**Pressure:** {top20_selected['pressure_tag']} ({top20_selected['pressure_label']})")

    show_cols = [
        "rank","number","seta","type","pressure_tag","pressure_label",
        "x_start","y_start","x_end","y_end","xt_start","xt_end","deltaxT","deltaxT_final","pass_distance"
    ]
    st.dataframe(
        top20[show_cols].style.format({
            "x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}","y_end":"{:.2f}",
            "xt_start":"{:.4f}","xt_end":"{:.4f}",
            "deltaxT":"{:.4f}","deltaxT_final":"{:.4f}",
            "pass_distance":"{:.1f}"
        }),
        use_container_width=True,
        height=420
    )
