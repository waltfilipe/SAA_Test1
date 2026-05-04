import streamlit as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mplsoccer import Pitch

import pandas as pd
import numpy as np
import textwrap

from PIL import Image
from io import BytesIO

from matplotlib.patches import FancyArrowPatch, Rectangle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D

from collections import defaultdict
import math


st.set_page_config(layout='wide', page_title='Action Map - Clean (Actions + xT + Pressão)')


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


st.title('Action Map - Clean (Actions + xT + Pressão)')


FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
NX, NY = 16, 12
LATERAL_MIN_DIST = 12.0

CMAP_ACTION = LinearSegmentedColormap.from_list(
    'xt_action',
    ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#f03b20','#bd0026','#67000d']
)
NORM_ACTION = Normalize(vmin=0.0, vmax=1.0)

# Distância (mantida)
D_REF = 10.0
D_SCALE = 20.0
BONUS_CAP = 0.60

# Pressão (novo)
PRESSURE_FACTOR_MAP = {
    '0-0': 1.00,  # sem pressão
    '1-0': 0.85,  # pressão na origem
    '0-1': 0.90,  # pressão no destino
    '1-1': 0.75,  # pressão em ambos
}

FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

OFFSET_GRID_X = 10
OFFSET_GRID_Y = 8


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


# ========= DADOS (somente seus passes) =========
# Formato: (label, x_start, y_start, x_end, y_end, outcome, pressure_code, video)
matches_data = {
    '1-1': [
        ('Seta 1',27.09,76.41,56.84,66.76,'successful','1-1',None),
        ('Seta 1',13.95,77.07,36.89,76.57,'successful','1-1',None),
        ('Seta 4',36.23,77.24,40.38,78.73,'successful','1-1',None),
        ('Seta 10',25.26,77.07,39.05,75.74,'failed','1-1',None),  # PASSE ERRADO
    ],
    '0-0': [
        ('Seta 3',28.42,76.24,13.29,64.27,'successful','0-0',None),
        ('Seta 5',38.22,54.30,13.45,34.85,'successful','0-0',None),
        ('Seta 6',41.55,55.63,31.74,34.35,'successful','0-0',None),
        ('Seta 7',53.35,61.61,71.97,75.91,'successful','0-0',None),
        ('Seta 8',56.84,62.77,48.20,38.67,'successful','0-0',None),
    ],
    '1-0': [
        ('Seta 2',31.91,77.40,16.78,64.27,'successful','1-0',None),
        ('Seta 9',78.45,73.25,80.45,73.25,'successful','1-0',None),
        ('Seta 11',23.93,77.74,33.90,74.91,'failed','1-0',None),  # PASSE ERRADO
    ],
    '0-1': [
        ('Seta 1',21.10,71.42,46.70,70.09,'successful','0-1',None),
    ]
}


def has_video_value(v):
    return pd.notna(v) and str(v).strip() != ''


def classify_action_direction(x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    ang = np.degrees(np.arctan2(abs(dy), dx))
    if ang <= 45:
        return 'forward'
    if ang >= 135:
        return 'backward'
    return 'lateral' if dist > LATERAL_MIN_DIST else ('forward' if dx >= 0 else 'backward')


def recompute_bonus_and_pressure(df):
    df = df.copy()
    excess = np.maximum(0.0, df['action_distance'].values - D_REF)
    df['dist_bonus'] = np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

    df['pressure_factor'] = df['pressure_code'].map(PRESSURE_FACTOR_MAP).fillna(1.0).astype(float)

    # ΔxT base (com distância) — mesma filosofia anterior
    df['delta_xt_adj'] = np.where(df['outcome'] == 'successful', df['delta_xt'] * (1.0 + df['dist_bonus']), 0.0)

    # NOVO: ΔxT ajustado por pressão
    df['delta_xt_pressure'] = np.where(df['outcome'] == 'successful', df['delta_xt_adj'] * df['pressure_factor'], 0.0)
    return df


dfs_by_match = {}
for match_name, events in matches_data.items():
    dfm = pd.DataFrame(events, columns=['type','x_start','y_start','x_end','y_end','outcome','pressure_code','video'])
    dfm['match'] = match_name
    dfm['number'] = np.arange(1, len(dfm) + 1)
    dfm['is_won'] = dfm['outcome'].eq('successful')

    dfm['direction'] = dfm.apply(lambda r: classify_action_direction(r.x_start, r.y_start, r.x_end, r.y_end), axis=1)
    dfm['is_forward'] = dfm['direction'] == 'forward'
    dfm['is_backward'] = dfm['direction'] == 'backward'
    dfm['is_lateral'] = dfm['direction'] == 'lateral'

    dfm['xt_start'] = dfm.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    dfm['xt_end'] = dfm.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
    dfm['delta_xt'] = np.where(dfm['outcome']=='successful', dfm['xt_end'] - dfm['xt_start'], 0.0)

    dfm['action_distance'] = np.sqrt((dfm.x_end-dfm.x_start)**2 + (dfm.y_end-dfm.y_start)**2)
    dfm = recompute_bonus_and_pressure(dfm)

    dfs_by_match[match_name] = dfm

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {'All Matches': df_all}
full_data.update(dfs_by_match)


def compute_stats(df):
    total = len(df)
    successful = int(df['is_won'].sum())
    accuracy = (successful / total * 100) if total else 0.0

    succ_mask = df['outcome'] == 'successful'
    sum_delta_xt = float(df.loc[succ_mask, 'delta_xt_pressure'].sum()) if succ_mask.any() else 0.0

    pos_mask = succ_mask & (df['delta_xt_pressure'] > 0)
    pos_count = int(pos_mask.sum())
    pos_sum = float(df.loc[pos_mask, 'delta_xt_pressure'].sum()) if pos_count else 0.0
    pos_mean = float(df.loc[pos_mask, 'delta_xt_pressure'].mean()) if pos_count else 0.0
    pos_pct = (pos_count / total * 100) if total else 0.0

    top10_df = (df.loc[pos_mask].sort_values('delta_xt_pressure', ascending=False).head(10)) if pos_count else pd.DataFrame()
    top10_sum = float(top10_df['delta_xt_pressure'].sum()) if not top10_df.empty else 0.0
    top10_mean = float(top10_df['delta_xt_pressure'].mean()) if not top10_df.empty else 0.0

    xt_end_mean = float(df.loc[succ_mask, 'xt_end'].mean()) if succ_mask.any() else 0.0
    xt_end_sum = float(df.loc[succ_mask, 'xt_end'].sum()) if succ_mask.any() else 0.0

    failed_mask = df['outcome'] == 'failed'
    failed_count = int(failed_mask.sum())
    failed_xt_inv = (1.0 - df.loc[failed_mask, 'xt_end']) if failed_count else pd.Series([], dtype=float)
    failed_xt_sum = float(failed_xt_inv.sum()) if failed_count else 0.0
    failed_xt_mean = float(failed_xt_inv.mean()) if failed_count else 0.0

    pressure_mean = float(df['pressure_factor'].mean()) if total else 0.0
    sum_delta_xt_base = float(df.loc[succ_mask, 'delta_xt_adj'].sum()) if succ_mask.any() else 0.0

    return {
        'total_actions': total,
        'successful_actions': successful,
        'accuracy_pct': round(accuracy, 2),
        'forward_total': int(df['is_forward'].sum()),
        'backward_total': int(df['is_backward'].sum()),
        'lateral_total': int(df['is_lateral'].sum()),
        'sum_delta_xt': round(sum_delta_xt, 4),  # agora é com pressão
        'sum_delta_xt_base': round(sum_delta_xt_base, 4),
        'pressure_mean': round(pressure_mean, 4),
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


def compute_parallel_offsets(df, offset_step=1.5):
    n = len(df)
    xs0 = df['x_start'].values.copy().astype(float)
    ys0 = df['y_start'].values.copy().astype(float)
    xs1 = df['x_end'].values.copy().astype(float)
    ys1 = df['y_end'].values.copy().astype(float)

    if n == 0:
        return xs0, ys0, xs1, ys1

    sx = np.clip((xs0 / FIELD_X * OFFSET_GRID_X).astype(int), 0, OFFSET_GRID_X - 1)
    sy = np.clip((ys0 / FIELD_Y * OFFSET_GRID_Y).astype(int), 0, OFFSET_GRID_Y - 1)
    ex = np.clip((xs1 / FIELD_X * OFFSET_GRID_X).astype(int), 0, OFFSET_GRID_X - 1)
    ey = np.clip((ys1 / FIELD_Y * OFFSET_GRID_Y).astype(int), 0, OFFSET_GRID_Y - 1)

    groups = defaultdict(list)
    for i in range(n):
        groups[(sx[i], sy[i], ex[i], ey[i])].append(i)

    for _, idxs in groups.items():
        m = len(idxs)
        if m == 1:
            continue

        mean_dx = float(np.mean(xs1[idxs] - xs0[idxs]))
        mean_dy = float(np.mean(ys1[idxs] - ys0[idxs]))
        length = math.hypot(mean_dx, mean_dy)

        if length < 1e-6:
            perp_x, perp_y = 0.0, 1.0
        else:
            perp_x = -mean_dy / length
            perp_y = mean_dx / length

        half = (m - 1) / 2.0
        offsets = [(j - half) * offset_step for j in range(m)]

        for j, i in enumerate(idxs):
            d = offsets[j]
            xs0[i] += perp_x * d
            ys0[i] += perp_y * d
            xs1[i] += perp_x * d
            ys1[i] += perp_y * d

    return xs0, ys0, xs1, ys1


def _draw_comet(ax, x0, y0, x1, y1, color, alpha, lw_scale, seg_alpha_factors=None):
    segs = 10
    ts = np.linspace(0.0, 1.0, segs + 1)
    if seg_alpha_factors is None:
        seg_alpha_factors = np.ones(segs, dtype=float)
    for i in range(segs):
        t0 = ts[i]
        t1 = ts[i + 1]
        xa = x0 + (x1 - x0) * t0
        ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1
        yb = y0 + (y1 - y0) * t1
        seg_alpha = alpha * (0.12 + 0.88 * t1) * float(seg_alpha_factors[i])
        seg_lw = 0.90 + lw_scale * t1
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=seg_lw, alpha=seg_alpha,
                zorder=4, solid_capstyle='round')
    start_size = 10.0 + 5.0 * alpha
    ax.scatter([x0], [y0], s=start_size, marker='o', c=[color],
               edgecolors='white', linewidths=0.35, alpha=max(0.12, alpha * 0.75), zorder=6)
    end_size = 34.0 + 18.0 * alpha
    ax.scatter([x1], [y1], s=end_size, marker='h', c=[color],
               edgecolors='white', linewidths=0.45, alpha=min(1.0, alpha + 0.15), zorder=7)


def _segment_list(x0, y0, x1, y1, segs=10):
    ts = np.linspace(0.0, 1.0, segs + 1)
    out = []
    for i in range(segs):
        t0 = ts[i]
        t1 = ts[i + 1]
        xa = x0 + (x1 - x0) * t0
        ya = y0 + (y1 - y0) * t0
        xb = x0 + (x1 - x0) * t1
        yb = y0 + (y1 - y0) * t1
        out.append((xa, ya, xb, yb))
    return out


def _seg_intersect(a, b, c, d, eps=1e-9):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    def on_seg(p, q, r):
        return min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and on_seg(a, c, b):
        return True
    if abs(o2) <= eps and on_seg(a, d, b):
        return True
    if abs(o3) <= eps and on_seg(c, a, d):
        return True
    if abs(o4) <= eps and on_seg(c, b, d):
        return True
    return False


def _action_visual(row, pos_ref):
    if not row['is_won']:
        return '#aab2be', 0.22, 1.55, 0

    dxt = float(row['delta_xt_pressure'])  # ajustado por pressão
    if dxt <= 0.0:
        return '#ffd64d', 0.08, 1.30, 1

    rel = float(np.clip(dxt / (pos_ref + 1e-9), 0.0, 1.0))
    color = matplotlib.colors.to_hex(plt.cm.YlOrRd(0.28 + 0.72 * rel))
    alpha = 0.34 + 0.62 * rel
    lw_scale = 1.70 + 2.10 * rel
    return color, alpha, lw_scale, 2


def draw_action_map(df, title, top_n_highlight=20, offset_step=1.5):
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1a1a2e', line_color='#ffffff', line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(9.6, 7.2))
    fig.set_facecolor('#1a1a2e')
    fig.set_dpi(150)
    ax.axvline(x=HALF_LINE_X, color='#ffffff', lw=0.6, alpha=0.12, linestyle='--')

    xs0_off, ys0_off, xs1_off, ys1_off = compute_parallel_offsets(df, offset_step=offset_step)
    pos_vals = df.loc[(df['outcome'] == 'successful') & (df['delta_xt_pressure'] > 0), 'delta_xt_pressure'].to_numpy()
    pos_ref = float(np.percentile(pos_vals, 90)) if pos_vals.size else 1.0
    pos_ref = max(pos_ref, 1e-6)

    scores = np.where(df['is_won'].to_numpy(), df['delta_xt_pressure'].to_numpy(dtype=float), -1e9)
    seg_lists = [
        _segment_list(xs0_off[i], ys0_off[i], xs1_off[i], ys1_off[i], segs=10)
        for i in range(len(df))
    ]
    seg_alpha_factors = [np.ones(10, dtype=float) for _ in range(len(df))]
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j or scores[j] <= scores[i]:
                continue
            for si, s1 in enumerate(seg_lists[i]):
                if seg_alpha_factors[i][si] < 0.35:
                    continue
                a = (s1[0], s1[1])
                b = (s1[2], s1[3])
                for s2 in seg_lists[j]:
                    c = (s2[0], s2[1])
                    d = (s2[2], s2[3])
                    if _seg_intersect(a, b, c, d):
                        seg_alpha_factors[i][si] = 0.22
                        break

    def draw_row(row, pos):
        color, alpha, lw_scale, layer = _action_visual(row, pos_ref)
        ox0, oy0 = xs0_off[pos], ys0_off[pos]
        ox1, oy1 = xs1_off[pos], ys1_off[pos]

        _draw_comet(
            ax,
            ox0,
            oy0,
            ox1,
            oy1,
            color=color,
            alpha=alpha,
            lw_scale=lw_scale,
            seg_alpha_factors=seg_alpha_factors[pos],
        )
        return layer

    draw_order = sorted(range(len(df)), key=lambda i: (int(_action_visual(df.iloc[i], pos_ref)[3]), float(scores[i])))
    rows = list(df.iterrows())
    for pos in draw_order:
        _, row = rows[pos]
        draw_row(row, pos)

    ax.set_title(title, fontsize=12, color='#ffffff', pad=8)

    legend_items = [
        Line2D([0], [0], color=matplotlib.colors.to_hex(plt.cm.YlOrRd(0.90)), lw=2.8,
               marker='h', markersize=9, markerfacecolor=matplotlib.colors.to_hex(plt.cm.YlOrRd(0.90)),
               markeredgecolor='white', markeredgewidth=0.6, alpha=0.96,
               label='Successful (+ΔxT pressão)'),
        Line2D([0], [0], color='#ffd64d', lw=2.3,
               marker='h', markersize=9, markerfacecolor='#ffd64d',
               markeredgecolor='white', markeredgewidth=0.6, alpha=0.92,
               label='Successful (≤0 ΔxT pressão)'),
        Line2D([0], [0], color='#aab2be', lw=2.3,
               marker='o', markersize=8, markerfacecolor='#aab2be',
               markeredgecolor='white', markeredgewidth=0.6, alpha=0.90,
               label='Failed'),
    ]

    legend = ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.095),
                       ncol=3, frameon=True, facecolor='#1a1a2e', edgecolor='#6b6b8f',
                       fontsize='small', labelspacing=0.45, borderpad=0.70, handletextpad=0.65,
                       columnspacing=1.8)
    for t in legend.get_texts():
        t.set_color('white')
    legend.get_frame().set_alpha(0.98)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=Normalize(vmin=0.0, vmax=pos_ref))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, shrink=0.72)
    cbar.set_label('ΔxT pressão', color='#ffe6bf', fontsize=9, labelpad=3)
    cbar.ax.yaxis.set_tick_params(color='#ffe6bf', labelsize=7)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffe6bf')

    fig.subplots_adjust(left=0.01, right=0.92, top=0.975, bottom=0.10)

    fig.canvas.draw()
    ax_pos = ax.get_position()
    cx = (ax_pos.x0 + ax_pos.x1) / 2
    strip_mid = ax_pos.y0 - 0.016
    fig.patches.append(FancyArrowPatch(
        (cx - 0.055, strip_mid), (cx + 0.055, strip_mid),
        transform=fig.transFigure, arrowstyle='-|>',
        mutation_scale=14, linewidth=1.9, color='#cccccc'))
    fig.text(cx, strip_mid - 0.008, 'Attack Direction',
             ha='center', va='top', transform=fig.transFigure,
             fontsize=9.5, color='#cccccc')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)

    return Image.open(buf), ax, fig


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
        axes, [origin_counts, dest_counts], [norm_origin, norm_dest], ['Origin', 'Destination']
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


def render_top10(df, title='Top 10 - ΔxT (Adj. + Pressão) and xT End'):
    st.markdown(f'<h4 style="color:#ffffff;margin:0 0 6px 0;">{title}</h4>', unsafe_allow_html=True)
    df_s = df[(df['outcome'] == 'successful') & (df['delta_xt_pressure'] > 0)]
    if df_s.empty:
        st.caption('No successful actions with positive ΔxT.')
        return

    top = df_s.sort_values('delta_xt_pressure', ascending=False).head(10).reset_index(drop=True)

    rows_html = ''
    for i, row in top.iterrows():
        rank = i + 1
        color = matplotlib.colors.to_hex(CMAP_ACTION(NORM_ACTION(float(row['xt_end']))))
        dot = (f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{color};'
               f'vertical-align:middle;margin-right:5px;border:1px solid rgba(255,255,255,.4);"></span>')
        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,.05);'>"
            f"<td style='color:#888;text-align:center;padding:4px 8px;font-size:11px;'>#{rank}</td>"
            f"<td style='color:#ccc;text-align:center;padding:4px 8px;font-size:11px;'>{row['type']}</td>"
            f"<td style='color:#ccc;text-align:center;padding:4px 8px;font-size:11px;'>{row['pressure_code']}</td>"
            f"<td style='color:#fff;text-align:right;padding:4px 8px;font-weight:700;font-size:12px;'>{row['delta_xt_pressure']:.4f}</td>"
            f"<td style='text-align:center;padding:4px 8px;'>{dot}<span style='color:#fff;font-size:12px;'>{row['xt_end']:.4f}</span></td>"
            '</tr>'
        )

    table_html = textwrap.dedent(f"""
    <table style='width:100%;border-collapse:collapse;background:rgba(255,255,255,.03);border-radius:8px;overflow:hidden;margin-bottom:6px;'>
        <thead>
            <tr style='background:rgba(255,255,255,.06);'>
                <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>Rank</th>
                <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>Seta</th>
                <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>Pressão</th>
                <th style='color:#aaa;padding:5px 8px;text-align:right;font-weight:500;font-size:11px;'>ΔxT (Adj.+P)</th>
                <th style='color:#aaa;padding:5px 8px;text-align:center;font-weight:500;font-size:11px;'>xT End</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """)

    st.markdown(table_html, unsafe_allow_html=True)


def build_match_metrics(dfs_by_match):
    rows = []
    for match_name, dfm in dfs_by_match.items():
        dfr = recompute_bonus_and_pressure(dfm.copy())
        s = compute_stats(dfr)
        rows.append({
            'match': match_name,
            'sum_delta_xt': s['sum_delta_xt'],          # com pressão
            'sum_delta_xt_base': s['sum_delta_xt_base'],# sem pressão
            'pos_pct': s['pos_pct'],
            'top10_sum': s['top10_sum'],
            'top10_mean': s['top10_mean'],
            'xt_end_sum': s['xt_end_sum'],
            'xt_end_mean': s['xt_end_mean'],
            'failed_xt_sum': s['failed_xt_sum'],
            'failed_xt_mean': s['failed_xt_mean'],
            'pressure_mean': s['pressure_mean'],
        })
    return pd.DataFrame(rows)


def render_direction_cards(stats):
    cards = [
        ('Forward', '&rarr;', int(stats['forward_total']), '#34d399'),
        ('Backward', '&larr;', int(stats['backward_total']), '#f97316'),
        ('Lateral', '&harr;', int(stats['lateral_total']), '#60a5fa'),
    ]
    html = '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;">'
    for label, arrow, value, color in cards:
        html += (
            '<div style="padding:2px 2px;">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'font-size:11px;color:#cbd5e1;"><span>{label}</span><span style="color:{color};font-size:14px;">{arrow}</span></div>'
            f'<div style="font-size:18px;font-weight:700;color:#ffffff;line-height:1.1;">{value}</div>'
            '</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def plot_metric_line(metrics_df, key, label):
    if metrics_df.empty:
        st.info('No data available for this chart.')
        return

    y = metrics_df[key].astype(float).to_numpy()
    x = np.arange(len(y), dtype=float)
    labels = metrics_df['match'].tolist()

    if len(x) > 1:
        x_dense = np.linspace(x.min(), x.max(), max(260, len(x) * 40))
        y_dense = np.interp(x_dense, x, y)
        if len(x) > 2:
            kernel = np.array([1, 2, 3, 2, 1], dtype=float)
            kernel /= kernel.sum()
            y_dense = np.convolve(y_dense, kernel, mode='same')
    else:
        x_dense, y_dense = x, y

    base = min(0.0, float(y.min()))

    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=220)
    fig.patch.set_facecolor('#0b1220')
    ax.set_facecolor('#101827')

    ax.fill_between(x_dense, y_dense, base, color='#0ea5e9', alpha=0.18, zorder=1)
    ax.plot(x_dense, y_dense, color='#7dd3fc', linewidth=9, alpha=0.08, solid_capstyle='round', zorder=2)
    ax.plot(x_dense, y_dense, color='#38bdf8', linewidth=3.2, solid_capstyle='round', zorder=3)
    ax.scatter(x, y, s=62, color='#22d3ee', edgecolors='white', linewidths=1.0, zorder=4)

    y_avg = float(np.mean(y))
    ax.axhline(y_avg, color='#fbbf24', linestyle=(0, (4, 4)), linewidth=1.2, alpha=0.75, zorder=2)
    ax.text(
        x.max() + 0.03, y_avg, f' avg: {y_avg:.2f}',
        color='#fcd34d', fontsize=8, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', fc='#1f2937', ec='none', alpha=0.8),
    )

    for xi, yi in zip(x, y):
        ax.text(
            xi, yi, f'{yi:.2f}',
            color='#e2e8f0', fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.18', fc='#0f172a', ec='none', alpha=0.72),
            zorder=5,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha='right', color='#cbd5e1', fontsize=9)
    ax.tick_params(axis='y', colors='#cbd5e1', labelsize=9)
    ax.set_ylabel(label, color='#bfdbfe', fontsize=10)
    ax.set_title(f'Match-by-Match Trend - {label}', loc='left', color='#f8fafc', fontsize=14, fontweight='700', pad=12)
    ax.grid(axis='y', color='#94a3b8', alpha=0.24, linestyle='--', linewidth=0.8)
    ax.grid(axis='x', color='#94a3b8', alpha=0.12, linestyle=':', linewidth=0.6)
    ax.margins(x=0.03)
    fig.tight_layout(pad=1.4)

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


for key, default in [
    ('heat_selection', None),
    ('last_match', 'All Matches'),
    ('last_filter', 'All Actions'),
    ('selected_action', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


tab_maps, tab_stats = st.tabs(['Maps', 'Stats'])

with tab_maps:
    col_filters, col_main = st.columns([0.95, 3.35], gap='large')

    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

        st.markdown('### Match Selection')
        selected_match = st.selectbox('Choose the match', list(full_data.keys()), index=0)

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        st.markdown('### Pressure Filter')
        pressure_filter = st.multiselect(
            'Pressure code',
            ['0-0', '1-0', '0-1', '1-1'],
            default=['0-0', '1-0', '0-1', '1-1']
        )

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        st.markdown('### Action Filter')
        action_filter = st.radio('Filter actions to display', [
            'All Actions',
            'Top N Actions (ΔxT pressão)',
            'Unsuccessful Actions',
            'Successful Actions',
            'Positive xT only',
        ], index=0)
        top_n = st.number_input('Top N', min_value=1, max_value=100, value=20, step=1)

        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state['last_match'] != selected_match:
        st.session_state['heat_selection'] = None
        st.session_state['last_match'] = selected_match
    if st.session_state['last_filter'] != action_filter:
        st.session_state['heat_selection'] = None
        st.session_state['last_filter'] = action_filter

    df_base = recompute_bonus_and_pressure(full_data[selected_match].copy())
    df_base = df_base[df_base['pressure_code'].isin(pressure_filter)].reset_index(drop=True)

    if action_filter == 'All Actions':
        df_base = df_base.reset_index(drop=True)
    elif action_filter == 'Top N Actions (ΔxT pressão)':
        df_s = df_base[df_base['outcome'] == 'successful']
        df_base = df_s.sort_values('delta_xt_pressure', ascending=False).head(int(top_n)).reset_index(drop=True)
    elif action_filter == 'Unsuccessful Actions':
        df_base = df_base[df_base['outcome'] == 'failed'].reset_index(drop=True)
    elif action_filter == 'Successful Actions':
        df_base = df_base[df_base['outcome'] == 'successful'].reset_index(drop=True)
    elif action_filter == 'Positive xT only':
        df_base = df_base[(df_base['outcome'] == 'successful') & (df_base['delta_xt_pressure'] > 0)].reset_index(drop=True)

    with col_main:
        DISPLAY_WIDTH = 1120
        df_to_draw = df_base

        st.markdown('<h4 style="color:#ffffff;margin:4px 0 3px 0;">Action Map</h4>', unsafe_allow_html=True)
        img_obj, ax, fig = draw_action_map(df_to_draw, title=f'Action Map - {selected_match}', top_n_highlight=int(top_n), offset_step=1.5)
        click = streamlit_image_coordinates(img_obj, width=DISPLAY_WIDTH)

        if click is not None and len(df_to_draw) > 0:
            rw, rh = img_obj.size
            px = click['x'] * (rw / click['width'])
            py = click['y'] * (rh / click['height'])
            fx, fy = ax.transData.inverted().transform((px, rh - py))
            df_sel = df_to_draw.copy()
            df_sel['dist'] = np.sqrt((df_sel.x_start - fx)**2 + (df_sel.y_start - fy)**2)
            cands = df_sel[df_sel['dist'] < 5.0]
            if not cands.empty:
                st.session_state['selected_action'] = cands.sort_values('dist').iloc[0]

        plt.close(fig)

        st.markdown('<h4 style="color:#ffffff;margin:6px 0 4px 0;">Event Panel</h4>', unsafe_allow_html=True)
        selected_action = st.session_state.get('selected_action', None)
        if selected_action is None:
            st.info('Click the origin ring on the map to open the event.')
        else:
            act_color = matplotlib.colors.to_hex(CMAP_ACTION(NORM_ACTION(float(selected_action['xt_end']))))
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
                f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;background:{act_color};border:2px solid #fff;"></span>'
                f'<strong style="color:#fff;">{selected_action["type"]} - {selected_action["outcome"].upper()}</strong></div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)
            with c1:
                st.write(f'**Start:** ({selected_action.x_start:.2f}, {selected_action.y_start:.2f})')
                st.write(f'**End:** ({selected_action.x_end:.2f}, {selected_action.y_end:.2f})')
                st.write(f'**Direction:** {selected_action["direction"].capitalize()}')
                st.write(f'**Pressure:** {selected_action["pressure_code"]}')
                st.write(f'**Successful:** {"Yes" if selected_action["is_won"] else "No"}')
            with c2:
                st.metric('Distance', f'{selected_action["action_distance"]:.1f}m')
                st.metric('ΔxT base', f'{selected_action["delta_xt_adj"]:.4f}')
                st.metric('ΔxT pressão', f'{selected_action["delta_xt_pressure"]:.4f}')

            if has_video_value(selected_action['video']):
                try:
                    st.video(selected_action['video'])
                except Exception:
                    st.error(f'Video not found: {selected_action["video"]}')
            else:
                st.warning('No video attached to this event.')

        st.markdown('<h4 style="color:#ffffff;margin:8px 0 4px 0;">Zone Heatmaps</h4>', unsafe_allow_html=True)
        hm_panel_img, _, hm_panel_fig = draw_zone_heatmaps_panel(df_base)
        st.image(hm_panel_img, use_column_width=True)
        plt.close(hm_panel_fig)

        st.markdown('<h4 style="color:#ffffff;margin:8px 0 4px 0;">Mini Maps - Top Zone Connections</h4>', unsafe_allow_html=True)
        mini_img, _, mini_fig = draw_top_connection_minimaps(df_base, top_k=3)
        st.image(mini_img, use_column_width=True)
        plt.close(mini_fig)


with tab_stats:
    st.subheader('Stats (General, Advanced)')

    selected_match_stats = st.selectbox('Match for Stats', list(full_data.keys()), index=0, key='stats_match_select')
    stats_df = recompute_bonus_and_pressure(full_data[selected_match_stats].copy())
    stats = compute_stats(stats_df)

    col_left, col_right = st.columns([1.02, 1.25], gap='large')

    with col_left:
        render_top10(stats_df, title='Top 10 ΔxT (Adj. + Pressão)')

    with col_right:
        with st.expander('General Statistics', expanded=True):
            st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1: small_metric('Total Actions', f"{stats['total_actions']}")
            with r2: small_metric('Successful', f"{stats['successful_actions']}")
            with r3: small_metric('Accuracy', f"{stats['accuracy_pct']:.1f}%")

            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="stats-section-title">Directions</div>', unsafe_allow_html=True)
            render_direction_cards(stats)

        with st.expander('Advanced Statistics', expanded=True):
            st.markdown('<div class="stats-section-title">ΔxT</div>', unsafe_allow_html=True)
            a1, a2, a3 = st.columns(3)
            with a1: small_metric('Σ ΔxT Base', f"{stats['sum_delta_xt_base']:.2f}")
            with a2: small_metric('Σ ΔxT Pressão', f"{stats['sum_delta_xt']:.2f}")
            with a3: small_metric('Fator pressão médio', f"{stats['pressure_mean']:.2f}")

            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="stats-section-title">Top 10</div>', unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            with t1: small_metric('Σ Top10', f"{stats['top10_sum']:.2f}")
            with t2: small_metric('Avg. Top10', f"{stats['top10_mean']:.2f}")

            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="stats-section-title">End xT</div>', unsafe_allow_html=True)
            e1, e2 = st.columns(2)
            with e1: small_metric('Σ End xT', f"{stats['xt_end_sum']:.2f}")
            with e2: small_metric('Avg. End xT', f"{stats['xt_end_mean']:.2f}")

            st.markdown('<hr style="margin:6px 0 8px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="stats-section-title">Failed</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1: small_metric('Σ xT End (Failed)', f"{stats['failed_xt_sum']:.2f}")
            with f2: small_metric('Avg. xT (Failed)', f"{stats['failed_xt_mean']:.2f}")

    st.markdown('<h4 style="color:#ffffff;margin:12px 0 6px 0;">Match Trend Line Chart</h4>', unsafe_allow_html=True)

    metric_options = {
        'Σ ΔxT Pressão': ('sum_delta_xt', 'Σ ΔxT Pressão'),
        'Σ ΔxT Base': ('sum_delta_xt_base', 'Σ ΔxT Base'),
        '% Positive ΔxT': ('pos_pct', '% Positive ΔxT'),
        'Σ Top 10 ΔxT': ('top10_sum', 'Σ Top 10 ΔxT'),
        'Avg. Top 10 ΔxT': ('top10_mean', 'Avg. Top 10 ΔxT'),
        'Σ End xT': ('xt_end_sum', 'Σ End xT'),
        'Avg. End xT': ('xt_end_mean', 'Avg. End xT'),
        'Σ xT End (Failed)': ('failed_xt_sum', 'Σ xT End (Failed)'),
        'Avg. xT (Failed)': ('failed_xt_mean', 'Avg. xT (Failed)'),
    }

    metric_label = st.selectbox('Select a metric for the chart', list(metric_options.keys()), index=0)
    metric_key, metric_name = metric_options[metric_label]
    match_metrics_df = build_match_metrics(dfs_by_match)
    plot_metric_line(match_metrics_df, metric_key, metric_name)

    st.caption(
        'Mapping: origin (ring), destination (diamond), color by xT End, '
        'and value ranking by ΔxT adjusted by distance and pressure.'
    )
