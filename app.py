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

st.set_page_config(layout="wide", page_title="Pass Map Dashboard — Pressão + xT")

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

st.title("Pass Map Dashboard — Pressão + xT")

# ── Constants ────────────────────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
LATERAL_MIN_DIST = 12.0

NX, NY = 16, 12
FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

# cores mantendo estética
COLOR_SUCCESS = "#c8c8c8"
COLOR_FAIL = "#E07070"
COLOR_XT_POS = "#2F80ED"
ALPHA_SUCCESS = 0.09

PRESSURE_COLORS = {
    "0-0": "#9CA3AF",  # sem pressão
    "1-0": "#F59E0B",  # pressão origem
    "0-1": "#06B6D4",  # pressão destino
    "1-1": "#8B5CF6",  # pressão ambos
}

D_REF = 10.0
D_SCALE = 20.0
BONUS_CAP = 0.60

# ── xT logic (adaptado da lógica do xT_actions_teste_v2) ─────────────────────
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
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x/FIELD_X)*NX, 0, NX-1))
    iy = int(np.clip((y/FIELD_Y)*NY, 0, NY-1))
    return float(XT_GRID[iy, ix])

def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

# ── Dados de pressão enviados por você ────────────────────────────────────────
# columns: label, x_start, y_start, x_end, y_end, pressure_tag, is_won
raw_pressure_passes = [
    # 1-1
    ("Seta 1", 27.09, 76.41, 56.84, 66.76, "1-1", True),
    ("Seta 1", 13.95, 77.07, 36.89, 76.57, "1-1", True),
    ("Seta 4", 36.23, 77.24, 40.38, 78.73, "1-1", True),
    ("Seta 10", 25.26, 77.07, 39.05, 75.74, "1-1", False),  # passe errado

    # 0-0
    ("Seta 3", 28.42, 76.24, 13.29, 64.27, "0-0", True),
    ("Seta 5", 38.22, 54.30, 13.45, 34.85, "0-0", True),
    ("Seta 6", 41.55, 55.63, 31.74, 34.35, "0-0", True),
    ("Seta 7", 53.35, 61.61, 71.97, 75.91, "0-0", True),
    ("Seta 8", 56.84, 62.77, 48.20, 38.67, "0-0", True),

    # 1-0
    ("Seta 2", 31.91, 77.40, 16.78, 64.27, "1-0", True),
    ("Seta 9", 78.45, 73.25, 80.45, 73.25, "1-0", True),
    ("Seta 11", 23.93, 77.74, 33.90, 74.91, "1-0", False),  # passe errado

    # 0-1
    ("Seta 1", 21.10, 71.42, 46.70, 70.09, "0-1", True),
]

pressure_label_map = {
    "0-0": "Sem pressão",
    "1-0": "Pressão na origem",
    "0-1": "Pressão no destino",
    "1-1": "Pressão em ambos",
}

df = pd.DataFrame(
    raw_pressure_passes,
    columns=["seta", "x_start", "y_start", "x_end", "y_end", "pressure_tag", "is_won"]
)
df["number"] = np.arange(1, len(df) + 1)
df["type"] = np.where(df["is_won"], "PASS WON", "PASS LOST")
df["outcome"] = np.where(df["is_won"], "completed", "incomplete")
df["pressure_label"] = df["pressure_tag"].map(pressure_label_map)

df["xt_start"] = df.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
df["xt_end"] = df.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
df["delta_xt_raw"] = np.where(df["is_won"], df["xt_end"] - df["xt_start"], 0.0)
df["pass_distance"] = np.sqrt((df.x_end-df.x_start)**2 + (df.y_end-df.y_start)**2)
df["dist_bonus"] = distance_bonus(df["pass_distance"].values)
df["delta_xt_adj"] = np.where(df["is_won"], df["delta_xt_raw"] * (1 + df["dist_bonus"]), 0.0)

# ── Draw helpers ──────────────────────────────────────────────────────────────
def _base_pitch():
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
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
    fig.text(0.5,0.02,"Attacking Direction",ha="center",va="center",fontsize=9,color="#cccccc")

def _save_fig(fig):
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df_plot: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()
    pos_ref = max(float(np.percentile(df_plot[df_plot["delta_xt_adj"] > 0]["delta_xt_adj"], 90))
                  if (df_plot["delta_xt_adj"] > 0).any() else 1.0, 1e-6)

    for _, row in df_plot.iterrows():
        pcolor = PRESSURE_COLORS.get(row["pressure_tag"], "#c8c8c8")
        if not row["is_won"]:
            color, alpha, lw = COLOR_FAIL, 0.82, 1.6
        else:
            rel = float(np.clip(row["delta_xt_adj"] / pos_ref, 0, 1))
            # mistura cor da pressão com intensidade de xT
            color = pcolor if row["delta_xt_adj"] <= 0 else COLOR_XT_POS
            alpha = 0.35 + 0.55 * rel if row["delta_xt_adj"] > 0 else 0.28
            lw = 1.5 + 1.8 * rel

        pitch.arrows(row.x_start,row.y_start,row.x_end,row.y_end,
                     color=color,width=lw,headwidth=2.3,headlength=2.3,ax=ax,zorder=3,alpha=alpha)
        pitch.scatter(row.x_start,row.y_start,s=45,marker="o",color=pcolor,
                      edgecolors="white",linewidths=0.8,ax=ax,zorder=6,alpha=0.95)

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)
    leg = ax.legend(handles=[
        Line2D([0],[0],color=PRESSURE_COLORS["0-0"], lw=2.5,label="0-0 Sem pressão", alpha=0.9),
        Line2D([0],[0],color=PRESSURE_COLORS["1-0"], lw=2.5,label="1-0 Pressão origem", alpha=0.9),
        Line2D([0],[0],color=PRESSURE_COLORS["0-1"], lw=2.5,label="0-1 Pressão destino", alpha=0.9),
        Line2D([0],[0],color=PRESSURE_COLORS["1-1"], lw=2.5,label="1-1 Pressão ambos", alpha=0.9),
        Line2D([0],[0],color=COLOR_XT_POS, lw=2.5,label="ΔxT positivo", alpha=0.9),
        Line2D([0],[0],color=COLOR_FAIL, lw=2.5,label="Passe errado", alpha=0.9),
    ], loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
       facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small", labelspacing=0.5, borderpad=0.5)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.92)
    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

def compute_stats(df_stats: pd.DataFrame):
    total = len(df_stats)
    comp = int(df_stats["is_won"].sum())
    acc = round(comp / max(total, 1) * 100, 2)

    succ = df_stats[df_stats["is_won"]]
    sum_xt = float(succ["delta_xt_adj"].sum()) if not succ.empty else 0.0
    mean_xt = float(succ["delta_xt_adj"].mean()) if not succ.empty else 0.0
    pos_xt = succ[succ["delta_xt_adj"] > 0]
    pos_pct = round(len(pos_xt) / max(total, 1) * 100, 2)

    by_pressure = (
        df_stats.groupby("pressure_tag", dropna=False)
        .agg(
            total=("number","count"),
            completed=("is_won","sum"),
            xt_sum=("delta_xt_adj","sum"),
            xt_mean=("delta_xt_adj","mean"),
        )
        .reset_index()
    )
    by_pressure["accuracy_pct"] = np.where(
        by_pressure["total"] > 0, by_pressure["completed"] / by_pressure["total"] * 100, 0
    )
    by_pressure["pressure_label"] = by_pressure["pressure_tag"].map(pressure_label_map)

    return {
        "total": total,
        "completed": comp,
        "accuracy": acc,
        "sum_xt": sum_xt,
        "mean_xt": mean_xt,
        "pos_pct": pos_pct,
        "by_pressure": by_pressure.sort_values("pressure_tag")
    }

# ── UI ────────────────────────────────────────────────────────────────────────
col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### ⚡ Pressão")
    pressure_filter = st.radio(
        "Filtro de pressão",
        ["Todos", "0-0", "1-0", "0-1", "1-1"],
        index=0
    )
    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
    st.markdown("### 🎯 Passe")
    pass_filter = st.radio(
        "Filtro de passe",
        ["Todos", "Completos", "Errados", "ΔxT positivo"],
        index=0
    )
    st.markdown("</div>", unsafe_allow_html=True)

df_base = df.copy()
if pressure_filter != "Todos":
    df_base = df_base[df_base["pressure_tag"] == pressure_filter].reset_index(drop=True)

if pass_filter == "Completos":
    df_base = df_base[df_base["is_won"]].reset_index(drop=True)
elif pass_filter == "Errados":
    df_base = df_base[~df_base["is_won"]].reset_index(drop=True)
elif pass_filter == "ΔxT positivo":
    df_base = df_base[(df_base["is_won"]) & (df_base["delta_xt_adj"] > 0)].reset_index(drop=True)

with col_field:
    st.caption("Clique no ponto de origem para inspecionar o passe.")
    img_obj, ax, fig = draw_pass_map(df_base, "Passes por Pressão + xT")
    click = streamlit_image_coordinates(img_obj, width=780, key="pm_map_pressure")

    selected_pass = None
    if click is not None and len(df_base) > 0:
        rw, rh = img_obj.size
        px = click["x"] * (rw / click["width"])
        py = click["y"] * (rh / click["height"])
        fx, fy = ax.transData.inverted().transform((px, rh - py))
        df_sel = df_base.copy()
        df_sel["_dist"] = np.sqrt((df_sel.x_start-fx)**2 + (df_sel.y_start-fy)**2)
        cands = df_sel[df_sel["_dist"] < 5.0].sort_values("_dist")
        if not cands.empty:
            selected_pass = cands.iloc[0]
    plt.close(fig)

    st.divider()
    st.subheader("Selected Event")
    if selected_pass is None:
        st.info("Clique em um passe no mapa para ver detalhes.")
    else:
        ok = "✅ Completo" if selected_pass["is_won"] else "❌ Errado"
        st.success(
            f"{selected_pass['seta']} — {ok} | {selected_pass['pressure_tag']} ({selected_pass['pressure_label']})"
        )
        c1, c2 = st.columns(2)
        c1.write(f"**Origem:** ({selected_pass.x_start:.2f}, {selected_pass.y_start:.2f})")
        c2.write(f"**Destino:** ({selected_pass.x_end:.2f}, {selected_pass.y_end:.2f})")
        c3, c4 = st.columns(2)
        c3.metric("xT início", f"{selected_pass.xt_start:.4f}")
        c4.metric("xT fim", f"{selected_pass.xt_end:.4f}")
        c5, c6 = st.columns(2)
        c5.metric("ΔxT ajustado", f"{selected_pass.delta_xt_adj:.4f}")
        c6.metric("Distância", f"{selected_pass.pass_distance:.1f} m")

    with st.expander("📊 Full Pass Data Table", expanded=False):
        cols = [
            "number","seta","type","pressure_tag","pressure_label","x_start","y_start","x_end","y_end",
            "xt_start","xt_end","delta_xt_raw","dist_bonus","delta_xt_adj","pass_distance"
        ]
        st.dataframe(
            df_base[cols].style.format({
                "x_start":"{:.2f}","y_start":"{:.2f}","x_end":"{:.2f}","y_end":"{:.2f}",
                "xt_start":"{:.4f}","xt_end":"{:.4f}","delta_xt_raw":"{:.4f}",
                "dist_bonus":"{:.3f}","delta_xt_adj":"{:.4f}","pass_distance":"{:.1f}"
            }),
            use_container_width=True,
            height=350
        )

with col_stats:
    s = compute_stats(df_base)

    with st.expander("📋 General Statistics", expanded=True):
        st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        with r1: small_metric("Total", f"{s['total']}")
        with r2: small_metric("Completos", f"{s['completed']}")
        with r3: small_metric("Acurácia", f"{s['accuracy']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="stats-section-title">xT</div>', unsafe_allow_html=True)
        x1,x2,x3 = st.columns(3)
        with x1: small_metric("Σ ΔxT", f"{s['sum_xt']:.3f}")
        with x2: small_metric("Média ΔxT", f"{s['mean_xt']:.3f}")
        with x3: small_metric("% ΔxT > 0", f"{s['pos_pct']:.1f}%")

    with st.expander("🧩 Por Cenário de Pressão", expanded=True):
        for _, row in s["by_pressure"].iterrows():
            st.markdown(f"<div class='stats-section-title'>{row['pressure_tag']} — {row['pressure_label']}</div>",
                        unsafe_allow_html=True)
            a,b,c = st.columns(3)
            with a: small_metric("Total", f"{int(row['total'])}")
            with b: small_metric("Acurácia", f"{row['accuracy_pct']:.1f}%")
            with c: small_metric("Σ ΔxT", f"{float(row['xt_sum']):.3f}")
            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)

    st.caption(
        "0-0 = sem pressão · 1-0 = pressão na origem · 0-1 = pressão no destino · 1-1 = pressão em ambos"
    )
