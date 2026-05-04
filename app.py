import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict
import math

st.set_page_config(layout="wide", page_title="Action Map - Caleb Simmons")

st.markdown('''
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
.performance-card{
    background:linear-gradient(150deg,rgba(14,165,233,.18) 0%,rgba(56,189,248,.10) 50%,rgba(12,74,110,.20) 100%);
    border:1px solid rgba(125,211,252,.28);
    border-radius:14px;
    padding:16px 14px;
    box-shadow:0 6px 24px rgba(2,6,23,.35);
}
</style>
''', unsafe_allow_html=True)

def small_metric(label, value, delta=None):
    html = (f'<div class="small-metric">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>')
    if delta is not None:
        html += f'<div class="delta">{delta}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

st.title('Action Map - Caleb Simmons')

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
NX, NY = 16, 12
LATERAL_MIN_DIST = 12.0

D_REF = 10.0
D_SCALE = 20.0
BONUS_CAP = 0.60

FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

CMAP_TOP15 = LinearSegmentedColormap.from_list(
    "top15_scale", ["#FDE047", "#FB923C", "#7F1D1D"]  # yellow -> orange -> dark red
)
NORM_TOP15 = Normalize(vmin=0.1, vmax=0.5)

# -----------------------------------------------------------------------------
# xT model
# -----------------------------------------------------------------------------
def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

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
        p = np.pad(a, ((ry,ry),(rx,rx)), mode='edge').astype(np.float64)
        ii = p.cumsum(0).cumsum(1)
        s = ii[2*ry:2*ry+H, 2*rx:2*rx+W].copy()
        s += ii[:H,:W]
        s -= ii[:H,2*rx:2*rx+W]
        s -= ii[2*ry:2*ry+H,:W]
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
    return XTc, XT

XT_GRID, _ = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x/FIELD_X)*NX, 0, NX-1))
    iy = int(np.clip((y/FIELD_Y)*NY, 0, NY-1))
    return float(XT_GRID[iy, ix])

# -----------------------------------------------------------------------------
# Data by match (from your message)
# -----------------------------------------------------------------------------
matches_data = {
    "Vs Connecticut": [
        # Completed
        ('ACTION WON',26.75,68.34,8.97,51.05,None),('ACTION WON',31.24,51.22,34.57,72.50,None),
        ('ACTION WON',36.06,46.90,44.37,57.04,None),('ACTION WON',48.36,64.02,58.17,51.72,None),
        ('ACTION WON',58.17,64.02,62.49,55.21,None),('ACTION WON',54.51,49.72,64.82,61.69,None),
        ('ACTION WON',42.21,70.84,34.90,76.49,None),('ACTION WON',43.54,75.32,36.73,67.84,None),
        ('ACTION WON',32.24,53.96,6.81,38.50,None),('ACTION WON',33.57,65.77,36.56,75.57,None),
        ('ACTION WON',37.39,61.11,43.04,75.41,None),('ACTION WON',65.49,53.63,56.18,70.42,None),
        ('ACTION WON',55.68,48.15,46.87,30.86,None),('ACTION WON',52.02,22.05,46.70,41.99,None),
        ('ACTION WON',62.16,35.51,71.80,35.18,None),('ACTION WON',54.02,33.35,63.99,22.55,None),
        ('ACTION WON',60.00,22.21,76.62,32.85,None),('ACTION WON',87.10,9.41,77.45,16.23,None),
        ('ACTION WON',62.66,20.05,117.18,8.25,None),('ACTION WON',98.90,43.49,103.22,47.15,None),
        ('ACTION WON',70.31,45.98,82.28,60.11,None),('ACTION WON',85.10,75.24,101.39,74.08,None),
        ('ACTION WON',53.18,67.59,39.05,59.62,None),('ACTION WON',55.18,49.64,54.85,13.07,None),
        ('ACTION WON',68.64,19.22,49.03,24.37,None),('ACTION WON',53.35,22.71,59.34,30.19,None),
        ('ACTION WON',44.37,24.71,40.05,46.82,None),('ACTION WON',43.88,39.34,41.38,73.08,None),
        ('ACTION WON',56.84,53.46,70.81,76.24,None),('ACTION WON',82.77,12.24,91.42,4.59,None),
        ('ACTION WON',108.04,11.74,115.69,58.29,None),('ACTION WON',93.08,3.93,111.03,13.74,None),
        ('ACTION WON',84.60,17.89,96.74,22.05,None),('ACTION WON',58.34,16.06,65.65,2.43,None),
        ('ACTION WON',52.02,8.58,44.37,15.73,None),('ACTION WON',61.00,23.21,49.36,15.23,None),
        ('ACTION WON',32.74,30.69,50.03,33.02,None),('ACTION WON',51.85,33.68,60.66,40.00,None),
        ('ACTION WON',79.95,60.45,98.23,60.28,None),('ACTION WON',31.24,52.14,39.05,72.08,None),
        ('ACTION WON',39.72,48.98,33.40,57.62,None),('ACTION WON',70.64,51.47,61.00,51.64,None),
        # Failed
        ('ACTION LOST',53.35,19.55,73.96,11.24,None),('ACTION LOST',63.82,20.55,88.76,22.55,None),
        ('ACTION LOST',85.60,27.86,94.41,37.17,None),('ACTION LOST',77.79,27.53,96.41,25.37,None),
        ('ACTION LOST',91.09,27.86,109.54,50.47,None),('ACTION LOST',58.17,26.04,95.41,40.33,None),
        ('ACTION LOST',53.35,28.53,73.80,27.86,None),('ACTION LOST',53.35,34.02,84.60,58.62,None),
        ('ACTION LOST',56.18,49.48,97.07,62.11,None),('ACTION LOST',34.23,74.91,65.65,78.57,None),
    ],
    "Vs Nashville": [
        # Completed
        ('ACTION WON',21.27,14.23,29.25,31.02,None),('ACTION WON',29.41,23.38,34.40,64.60,None),
        ('ACTION WON',41.55,39.67,41.88,6.92,None),('ACTION WON',44.54,32.52,43.54,14.23,None),
        ('ACTION WON',23.59,56.46,34.57,47.48,None),('ACTION WON',30.58,64.44,21.10,49.48,None),
        ('ACTION WON',33.07,56.79,49.53,69.59,None),('ACTION WON',33.24,59.78,44.04,71.75,None),
        ('ACTION WON',61.50,71.58,54.68,75.57,None),('ACTION WON',63.16,50.81,78.45,67.26,None),
        ('ACTION WON',63.49,76.90,84.44,62.77,None),('ACTION WON',76.96,56.96,86.93,57.79,None),
        ('ACTION WON',82.61,59.12,96.41,68.43,None),('ACTION WON',79.78,35.35,106.21,11.74,None),
        ('ACTION WON',45.37,49.64,40.72,32.02,None),
        # Failed
        ('ACTION LOST',78.62,64.94,96.57,67.10,None),('ACTION LOST',85.43,68.76,106.05,77.74,None),
    ],
    "Vs Seongnam": [
        # Completed
        ('ACTION WON',28.08,28.53,29.75,8.25,None),('ACTION WON',33.74,26.54,29.41,43.82,None),
        ('ACTION WON',28.08,47.15,31.57,64.60,None),('ACTION WON',39.39,43.82,51.69,53.46,None),
        ('ACTION WON',43.88,46.15,55.84,40.66,None),('ACTION WON',47.03,49.97,44.04,28.03,None),
        ('ACTION WON',47.53,50.81,71.97,33.18,None),('ACTION WON',67.65,52.63,64.32,33.85,None),
        ('ACTION WON',73.63,65.10,69.31,73.25,None),('ACTION WON',77.29,63.27,79.12,72.91,None),
        ('ACTION WON',81.61,56.62,93.91,73.75,None),('ACTION WON',86.43,66.43,81.78,54.96,None),
        ('ACTION WON',111.03,71.42,99.56,67.59,None),('ACTION WON',89.76,59.62,97.74,48.98,None),
        ('ACTION WON',88.43,52.47,96.41,74.24,None),('ACTION WON',87.93,50.97,77.12,27.70,None),
        ('ACTION WON',81.61,53.63,74.30,27.03,None),('ACTION WON',79.28,51.14,94.91,70.42,None),
        ('ACTION WON',52.85,32.85,65.49,25.37,None),('ACTION WON',82.77,33.18,69.31,47.65,None),
        # Failed
        ('ACTION LOST',72.14,16.56,78.45,1.60,None),('ACTION LOST',79.62,27.53,97.07,47.98,None),
        ('ACTION LOST',91.75,50.14,109.70,65.77,None),('ACTION LOST',96.41,56.79,107.04,67.26,None),
    ],
    "Vs Red Bull": [
        # Completed
        ('ACTION WON',39.39,19.39,52.35,4.76,None),('ACTION WON',63.82,7.92,72.63,1.43,None),
        ('ACTION WON',70.47,11.91,80.95,13.74,None),('ACTION WON',64.49,22.55,97.24,10.24,None),
        ('ACTION WON',32.07,35.51,43.04,28.20,None),('ACTION WON',53.52,46.32,54.02,33.68,None),
        ('ACTION WON',77.12,48.64,84.94,50.14,None),('ACTION WON',78.12,52.47,117.52,69.42,None),
        ('ACTION WON',88.76,65.93,97.40,76.74,None),('ACTION WON',82.61,69.26,86.60,77.40,None),
        ('ACTION WON',78.62,66.26,79.62,78.40,None),('ACTION WON',83.61,75.91,62.49,57.12,None),
        ('ACTION WON',34.40,50.14,88.76,75.41,None),('ACTION WON',56.68,64.27,78.29,64.27,None),
        ('ACTION WON',51.85,73.25,54.18,78.07,None),('ACTION WON',41.05,57.45,46.04,74.91,None),
        ('ACTION WON',37.39,60.61,41.71,73.91,None),('ACTION WON',30.41,63.44,36.89,77.40,None),
        ('ACTION WON',26.09,63.94,28.42,76.74,None),('ACTION WON',22.43,56.62,22.10,76.41,None),
        ('ACTION WON',33.90,64.77,25.42,73.58,None),
        # Failed
        ('ACTION LOST',41.88,42.49,56.18,52.97,None),('ACTION LOST',37.56,41.16,46.37,53.96,None),
        ('ACTION LOST',54.68,56.96,54.85,64.44,None),('ACTION LOST',51.69,68.43,66.15,76.57,None),
    ],
}

# -----------------------------------------------------------------------------
# Build DF
# -----------------------------------------------------------------------------
def classify_action_direction(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    ang = np.degrees(np.arctan2(abs(dy), dx))
    if ang <= 45:
        return 'forward'
    if ang >= 135:
        return 'backward'
    return 'lateral' if dist > LATERAL_MIN_DIST else ('forward' if dx >= 0 else 'backward')

def recompute_bonus(df):
    df = df.copy()
    excess = np.maximum(0.0, df['action_distance'].values - D_REF)
    df['dist_bonus'] = np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))
    df['delta_xt_adj'] = np.where(df['outcome'] == 'successful', df['delta_xt'] * (1.0 + df['dist_bonus']), 0.0)
    return df

dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=['type','x_start','y_start','x_end','y_end','video'])
    dfm['match'] = match_name
    dfm['number'] = np.arange(1, len(dfm) + 1)
    dfm['is_won'] = dfm['type'].str.contains('WON', case=False)
    dfm['outcome'] = np.where(dfm['is_won'], 'successful', 'failed')
    dfm['direction'] = dfm.apply(lambda r: classify_action_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1)
    dfm['is_forward'] = dfm['direction'] == 'forward'
    dfm['is_backward'] = dfm['direction'] == 'backward'
    dfm['is_lateral'] = dfm['direction'] == 'lateral'
    dfm['xt_start'] = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    dfm['xt_end'] = dfm.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
    dfm['delta_xt'] = np.where(dfm['outcome']=='successful', dfm['xt_end'] - dfm['xt_start'], 0.0)
    dfm['action_distance'] = np.sqrt((dfm.x_end-dfm.x_start)**2 + (dfm.y_end-dfm.y_start)**2)
    dfm['dist_bonus'] = distance_bonus(dfm['action_distance'].values)
    dfm['delta_xt_adj'] = np.where(dfm['outcome']=='successful', dfm['delta_xt'] * (1 + dfm['dist_bonus']), 0.0)
    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {'All Matches': df_all}
full_data.update(dfs_by_match)

# -----------------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------------
def compute_stats(df):
    total = len(df)
    successful = int(df['is_won'].sum())
    accuracy = (successful / total * 100) if total else 0.0

    succ_mask = df['outcome'] == 'successful'
    sum_delta_xt = float(df.loc[succ_mask, 'delta_xt_adj'].sum()) if succ_mask.any() else 0.0

    pos_mask = succ_mask & (df['delta_xt_adj'] > 0)
    pos_count = int(pos_mask.sum())
    pos_sum = float(df.loc[pos_mask, 'delta_xt_adj'].sum()) if pos_count else 0.0
    pos_mean = float(df.loc[pos_mask, 'delta_xt_adj'].mean()) if pos_count else 0.0
    pos_pct = (pos_count / total * 100) if total else 0.0

    top10_df = (df.loc[pos_mask].sort_values('delta_xt_adj', ascending=False).head(10)) if pos_count else pd.DataFrame()
    top10_sum = float(top10_df['delta_xt_adj'].sum()) if not top10_df.empty else 0.0
    top10_mean = float(top10_df['delta_xt_adj'].mean()) if not top10_df.empty else 0.0

    xt_end_mean = float(df.loc[succ_mask, 'xt_end'].mean()) if succ_mask.any() else 0.0
    xt_end_sum = float(df.loc[succ_mask, 'xt_end'].sum()) if succ_mask.any() else 0.0

    failed_mask = df['outcome'] == 'failed'
    failed_count = int(failed_mask.sum())
    failed_xt_inv = (1.0 - df.loc[failed_mask, 'xt_end']) if failed_count else pd.Series([], dtype=float)
    failed_xt_sum = float(failed_xt_inv.sum()) if failed_count else 0.0
    failed_xt_mean = float(failed_xt_inv.mean()) if failed_count else 0.0

    return {
        'total_actions': total,
        'successful_actions': successful,
        'accuracy_pct': round(accuracy, 2),
        'forward_total': int(df['is_forward'].sum()),
        'backward_total': int(df['is_backward'].sum()),
        'lateral_total': int(df['is_lateral'].sum()),
        'sum_delta_xt': round(sum_delta_xt, 4),
        'positive_xt_count': pos_count,
        'pos_sum': round(pos_sum, 4),
        'pos_mean': round(pos_mean, 4),
        'pos_pct': round(pos_pct, 2),
        'top10_sum': round(top10_sum, 4),
        'top10_mean': round(top10_mean, 4),
        'xt_end_mean': round(xt_end_mean, 4),
        'xt_end_sum': round(xt_end_sum, 4),
        'failed_count': failed_count,
        'failed_xt_sum': round(failed_xt_sum, 4),
        'failed_xt_mean': round(failed_xt_mean, 4),
    }

# -----------------------------------------------------------------------------
# Map and analysis visuals
# -----------------------------------------------------------------------------
def draw_top15_map(df, title):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(8.8, 6.4))
    fig.set_facecolor('#1a1a2e')
    fig.set_dpi(145)

    # Final-third line: very light white dashed
    ax.axvline(x=FINAL_THIRD_LINE_X, color='#ffffff', lw=1.0, alpha=0.18, linestyle='--')
    ax.axvline(x=HALF_LINE_X, color='#ffffff', lw=0.6, alpha=0.10, linestyle='--')

    top15 = (
        df[(df['outcome'] == 'successful') & (df['delta_xt_adj'] > 0)]
        .sort_values('delta_xt_adj', ascending=False)
        .head(15)
        .copy()
        .reset_index(drop=True)
    )

    if top15.empty:
        ax.set_title(title, fontsize=12, color='#ffffff', pad=8)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=145, facecolor=fig.get_facecolor(), bbox_inches='tight')
        buf.seek(0)
        return Image.open(buf), ax, fig, top15

    top15['rank'] = np.arange(1, len(top15) + 1)

    for _, row in top15.iterrows():
        val = float(row['delta_xt_adj'])
        color = CMAP_TOP15(NORM_TOP15(np.clip(val, 0.1, 0.5)))
        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=color,width=2.2,headwidth=2.65,headlength=2.65,
                     ax=ax,zorder=4,alpha=0.95)
        pitch.scatter(row.x_start,row.y_start,s=55,marker='o',color=color,
                      edgecolors='white',linewidths=0.85,ax=ax,zorder=6,alpha=0.98)

    ax.set_title(title, fontsize=12, color='#ffffff', pad=8)

    leg = ax.legend(handles=[
        Line2D([0],[0],color=CMAP_TOP15(NORM_TOP15(0.10)), lw=3.0, label='ΔxT 0.10'),
        Line2D([0],[0],color=CMAP_TOP15(NORM_TOP15(0.30)), lw=3.0, label='ΔxT 0.30'),
        Line2D([0],[0],color=CMAP_TOP15(NORM_TOP15(0.50)), lw=3.0, label='ΔxT 0.50'),
    ], loc='upper left', bbox_to_anchor=(0.01,0.99), frameon=True,
       facecolor='#1a1a2e', edgecolor='#444466', fontsize='x-small',
       labelspacing=0.5, borderpad=0.5)
    for t in leg.get_texts():
        t.set_color('white')
    leg.get_frame().set_alpha(0.92)

    # Attack arrow
    ax_pos = ax.get_position()
    cx = (ax_pos.x0 + ax_pos.x1) / 2
    strip_mid = ax_pos.y0 - 0.02
    fig.patches.append(FancyArrowPatch(
        (cx - 0.055, strip_mid), (cx + 0.055, strip_mid),
        transform=fig.transFigure, arrowstyle='-|>',
        mutation_scale=14, linewidth=1.9, color='#cccccc'))
    fig.text(cx, strip_mid - 0.008, 'Attack Direction',
             ha='center', va='top', transform=fig.transFigure,
             fontsize=9.2, color='#cccccc')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=145, facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf), ax, fig, top15


def _zone_bins():
    x_bins = np.linspace(0, FIELD_X, 7)
    y_bins = np.array([0.0, LANE_RIGHT_MAX, LANE_LEFT_MIN, FIELD_Y])
    return x_bins, y_bins

def _zone_counts(df_s, x_col, y_col):
    x_bins, y_bins = _zone_bins()
    counts = np.zeros((3, 6), dtype=int)
    if df_s.empty:
        return counts
    ix = np.clip(np.searchsorted(x_bins, df_s[x_col].to_numpy(), side='right') - 1, 0, 5)
    iy = np.clip(np.searchsorted(y_bins, df_s[y_col].to_numpy(), side='right') - 1, 0, 2)
    for cx, cy in zip(ix, iy):
        counts[cy, cx] += 1
    return counts

def draw_zone_heatmaps_panel(df, title='Zone Heatmaps - Origin and Destination'):
    df_s = df[df['is_won']].copy()
    x_bins, y_bins = _zone_bins()
    origin_counts = _zone_counts(df_s, 'x_start', 'y_start')
    dest_counts = _zone_counts(df_s, 'x_end', 'y_end')
    cmap_h = LinearSegmentedColormap.from_list('wr', ['#ffffff', '#ffecec', '#ffbfbf', '#ff8080', '#ff3b3b', '#ff0000'])
    norm_origin = Normalize(vmin=0, vmax=max(1, int(origin_counts.max())))
    norm_dest = Normalize(vmin=0, vmax=max(1, int(dest_counts.max())))
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2.9, FIG_H * 1.55), dpi=FIG_DPI)
    fig.set_facecolor('#1a1a2e')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.95)
    for ax, counts, norm_h, subtitle in zip(
        axes,
        [origin_counts, dest_counts],
        [norm_origin, norm_dest],
        ['Origin', 'Destination']
    ):
        pitch.draw(ax=ax)
        for row in range(3):
            for col in range(6):
                x0, x1 = x_bins[col], x_bins[col + 1]
                y0, y1 = y_bins[row], y_bins[row + 1]
                val = int(counts[row, col])
                if val == 0:
                    continue
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       facecolor=cmap_h(norm_h(val)), edgecolor=(1, 1, 1, 0.12),
                                       lw=0.6, alpha=0.92, zorder=2))
                vmax_local = max(1, int(counts.max()))
                ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(val),
                        ha='center', va='center', zorder=4, fontsize=11,
                        color='#ffffff' if val >= max(2, int(vmax_local * 0.35)) else '#1d1d1d',
                        fontweight='600')
        ax.set_title(subtitle, fontsize=15, color='#ffffff', pad=8, fontweight='700')
        ax.axhline(y=LANE_LEFT_MIN, color='#ffffff', lw=0.5, alpha=0.12, linestyle='--', zorder=3)
        ax.axhline(y=LANE_RIGHT_MAX, color='#ffffff', lw=0.5, alpha=0.12, linestyle='--', zorder=3)
    fig.suptitle(title, fontsize=18, color='#ffffff', y=0.995, fontweight='700')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf), axes, fig

def _top_zone_transitions(df_s, top_k=14):
    x_bins, y_bins = _zone_bins()
    if df_s.empty:
        return [], x_bins, y_bins
    sx = np.clip(np.searchsorted(x_bins, df_s['x_start'].to_numpy(), side='right') - 1, 0, 5)
    sy = np.clip(np.searchsorted(y_bins, df_s['y_start'].to_numpy(), side='right') - 1, 0, 2)
    ex = np.clip(np.searchsorted(x_bins, df_s['x_end'].to_numpy(), side='right') - 1, 0, 5)
    ey = np.clip(np.searchsorted(y_bins, df_s['y_end'].to_numpy(), side='right') - 1, 0, 2)
    transitions = defaultdict(int)
    for a, b, c, d in zip(sx, sy, ex, ey):
        if int(a) == int(c) and int(b) == int(d):
            continue
        transitions[(int(a), int(b), int(c), int(d))] += 1
    return sorted(transitions.items(), key=lambda kv: kv[1], reverse=True)[:top_k], x_bins, y_bins

def draw_top_connection_minimaps(df, top_k=3, title='Top Zone Connections (Mini Maps)'):
    df_s = df[df['is_won']].copy()
    links, x_bins, y_bins = _top_zone_transitions(df_s, top_k=top_k)
    fig, axes = plt.subplots(1, top_k, figsize=(FIG_W * 1.6, FIG_H * 0.80), dpi=FIG_DPI)
    if top_k == 1:
        axes = [axes]
    fig.set_facecolor('#1a1a2e')
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.90)
    x_cent = (x_bins[:-1] + x_bins[1:]) / 2.0
    y_cent = (y_bins[:-1] + y_bins[1:]) / 2.0
    max_cnt = max([v for _, v in links], default=1)
    for idx, ax in enumerate(axes):
        pitch.draw(ax=ax)
        if idx >= len(links):
            ax.set_title('No link', fontsize=9, color='#dbeafe', pad=4)
            continue
        (ix0, iy0, ix1, iy1), cnt = links[idx]
        x0, y0 = float(x_cent[ix0]), float(y_cent[iy0])
        x1, y1 = float(x_cent[ix1]), float(y_cent[iy1])
        rel = cnt / max_cnt
        color = plt.cm.Blues(0.40 + 0.55 * rel)
        lw = 1.2 + 4.2 * rel
        alpha = 0.30 + 0.60 * rel
        ax.add_patch(Rectangle((x_bins[ix0], y_bins[iy0]), x_bins[ix0 + 1] - x_bins[ix0], y_bins[iy0 + 1] - y_bins[iy0],
                               facecolor=(0.20, 0.45, 0.95, 0.16), edgecolor=(1, 1, 1, 0.15), lw=0.6, zorder=2))
        ax.add_patch(Rectangle((x_bins[ix1], y_bins[iy1]), x_bins[ix1 + 1] - x_bins[ix1], y_bins[iy1 + 1] - y_bins[iy1],
                               facecolor=(0.02, 0.70, 0.55, 0.16), edgecolor=(1, 1, 1, 0.15), lw=0.6, zorder=2))
        if ix0 == ix1 and iy0 == iy1:
            ax.scatter([x0], [y0], s=40 + 80 * rel, c=[color], marker='o', edgecolors='white', linewidths=0.5, alpha=alpha, zorder=5)
        else:
            rad = float(np.clip(0.10 * np.sign((ix1 - ix0) + 0.4 * (iy1 - iy0)), -0.30, 0.30))
            arrow = FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}',
                                    arrowstyle='-|>', mutation_scale=10 + 9 * rel,
                                    lw=lw, color=color, alpha=alpha, zorder=4)
            ax.add_patch(arrow)
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, f'{cnt}', color='#e5efff', fontsize=8,
                ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.16', fc=(0.06, 0.09, 0.14, 0.80), ec='none'))
        ax.set_title(f'#{idx + 1}  {cnt}x', fontsize=9, color='#dbeafe', pad=4)
    fig.suptitle(title, fontsize=11, color='#ffffff', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf), axes, fig

# -----------------------------------------------------------------------------
# Session-state
# -----------------------------------------------------------------------------
for key, default in [
    ('selected_action', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_map, tab_analysis = st.tabs(['Map', 'Analyses'])

# -----------------------------------------------------------------------------
# TAB MAP
# -----------------------------------------------------------------------------
with tab_map:
    col_filters, col_main = st.columns([0.95, 3.35], gap='large')

    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        st.markdown('### Match Selection')
        selected_match = st.selectbox('Choose match', list(full_data.keys()), index=0)
        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown('### Display')
        st.caption('Top 15 successful actions by ΔxT.')
        st.markdown('</div>', unsafe_allow_html=True)

    df_base = recompute_bonus(full_data[selected_match].copy())

    with col_main:
        map_col, right_col = st.columns([2.15, 1.0], gap='small')

        with map_col:
            st.markdown('<h4 style="color:#ffffff;margin:4px 0 3px 0;">Top 15 ΔxT Map</h4>', unsafe_allow_html=True)

            img_obj, ax, fig, top15_df = draw_top15_map(df_base, title=f"Top 15 ΔxT — {selected_match}")
            click = streamlit_image_coordinates(img_obj, width=900)
            st.image(img_obj, use_container_width=False)

            if click is not None and not top15_df.empty:
                rw, rh = img_obj.size
                px = click['x'] * (rw / click['width'])
                py = click['y'] * (rh / click['height'])
                fx, fy = ax.transData.inverted().transform((px, rh - py))
                tmp = top15_df.copy()
                tmp['_dist'] = np.sqrt((tmp.x_start - fx)**2 + (tmp.y_start - fy)**2)
                cands = tmp[tmp['_dist'] < 5.0].sort_values('_dist')
                if not cands.empty:
                    st.session_state['selected_action'] = cands.iloc[0]
            plt.close(fig)

            st.markdown('<h4 style="color:#ffffff;margin:10px 0 4px 0;">Video</h4>', unsafe_allow_html=True)
            sel = st.session_state.get('selected_action', None)
            if sel is None:
                st.info('Select an action from Top 15 table or click on a start marker in the map.')
            elif pd.notna(sel['video']) and str(sel['video']).strip() != '':
                try:
                    st.video(sel['video'])
                except Exception:
                    st.error('Video not found.')
            else:
                st.warning('No video available for this action.')

        with right_col:
            st.markdown('<h4 style="color:#ffffff;margin:4px 0 3px 0;">Top 15 ΔxT Table</h4>', unsafe_allow_html=True)
            if top15_df.empty:
                st.caption('No successful actions with positive ΔxT.')
            else:
                show_df = pd.DataFrame({
                    'Rank': top15_df['rank'].astype(int),
                    'Action #': top15_df['number'].astype(int),
                    'ΔxT': top15_df['delta_xt_adj'].map(lambda x: f'{x:.4f}'),
                    'xT End': top15_df['xt_end'].map(lambda x: f'{x:.4f}')
                })
                event = st.dataframe(
                    show_df, use_container_width=True, height=470, hide_index=True,
                    selection_mode='single-row', on_select='rerun', key='top15_table'
                )
                if hasattr(event, 'selection') and event.selection.rows:
                    idx = int(event.selection.rows[0])
                    if 0 <= idx < len(top15_df):
                        st.session_state['selected_action'] = top15_df.iloc[idx]

            st.markdown('<h4 style="color:#ffffff;margin:14px 0 4px 0;">Event Panel</h4>', unsafe_allow_html=True)
            sel = st.session_state.get('selected_action', None)
            if sel is None:
                st.info('Select an event in the table or map.')
            else:
                rank_val = int(sel['rank']) if 'rank' in sel and not pd.isna(sel['rank']) else None
                act_color = matplotlib.colors.to_hex(CMAP_TOP15(NORM_TOP15(np.clip(float(sel['delta_xt_adj']), 0.1, 0.5))))
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                    f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{act_color};border:2px solid #fff;"></span>'
                    f'<strong style="color:#fff;">'
                    f'Action #{int(sel["number"])} — {sel["type"]}{" | Rank #" + str(rank_val) if rank_val else ""}'
                    f'</strong></div>',
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.write(f'**Start:** ({sel["x_start"]:.2f}, {sel["y_start"]:.2f})')
                    st.write(f'**End:** ({sel["x_end"]:.2f}, {sel["y_end"]:.2f})')
                    st.write(f'**Direction:** {str(sel["direction"]).capitalize()}')
                with c2:
                    st.metric('Distance', f'{float(sel["action_distance"]):.1f} m')
                    st.metric('ΔxT', f'{float(sel["delta_xt_adj"]):.4f}')

# -----------------------------------------------------------------------------
# TAB ANALYSES
# -----------------------------------------------------------------------------
with tab_analysis:
    selected_match_stats = st.selectbox('Match for Analyses', list(full_data.keys()), index=0, key='analysis_match')
    stats_df = recompute_bonus(full_data[selected_match_stats].copy())
    stats = compute_stats(stats_df)

    # Stylish KPI blocks
    st.markdown('<div style="font-size:18px;font-weight:700;color:#e0f2fe;margin-bottom:8px;">Performance Overview</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f"""<div class="performance-card">
                <div class="stats-section-title">Accuracy</div>
                <div style="font-size:30px;font-weight:700;color:#ffffff;">{stats['accuracy_pct']:.1f}%</div>
                <div style="font-size:11px;color:#bae6fd;">{stats['successful_actions']} / {stats['total_actions']} successful</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""<div class="performance-card">
                <div class="stats-section-title">Σ ΔxT</div>
                <div style="font-size:30px;font-weight:700;color:#ffffff;">{stats['sum_delta_xt']:.2f}</div>
                <div style="font-size:11px;color:#bae6fd;">Avg. positive: {stats['pos_mean']:.2f}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""<div class="performance-card">
                <div class="stats-section-title">% Positive ΔxT</div>
                <div style="font-size:30px;font-weight:700;color:#ffffff;">{stats['pos_pct']:.1f}%</div>
                <div style="font-size:11px;color:#bae6fd;">Count: {stats['positive_xt_count']}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    with p1: small_metric('Σ Top 10 ΔxT', f"{stats['top10_sum']:.2f}")
    with p2: small_metric('Avg. Top 10 ΔxT', f"{stats['top10_mean']:.2f}")
    with p3: small_metric('Σ End xT', f"{stats['xt_end_sum']:.2f}")
    with p4: small_metric('Σ xT Failed', f"{stats['failed_xt_sum']:.2f}")

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="stats-section-title">Direction Profile</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1: small_metric('Forward', f"{stats['forward_total']}")
    with d2: small_metric('Backward', f"{stats['backward_total']}")
    with d3: small_metric('Lateral', f"{stats['lateral_total']}")

    # Heatmaps + zone connections below stats
    st.markdown('<h4 style="color:#ffffff;margin:18px 0 6px 0;">Zone Heatmaps</h4>', unsafe_allow_html=True)
    hm_panel_img, _, hm_panel_fig = draw_zone_heatmaps_panel(stats_df, title='Zone Heatmaps - Origin and Destination')
    st.image(hm_panel_img, use_container_width=True)
    plt.close(hm_panel_fig)

    st.markdown('<h4 style="color:#ffffff;margin:12px 0 6px 0;">Mini Maps - Top Zone Connections</h4>', unsafe_allow_html=True)
    mini_img, _, mini_fig = draw_top_connection_minimaps(stats_df, top_k=3)
    st.image(mini_img, use_container_width=True)
    plt.close(mini_fig)

    st.caption('Color scale for Top 15 map uses ΔxT from 0.10 to 0.50 (yellow → orange → dark red).')
