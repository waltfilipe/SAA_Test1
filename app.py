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
from matplotlib.patches import FancyArrowPatch
from streamlit_image_coordinates import streamlit_image_coordinates

# =============================================================================
# Page + Style
# =============================================================================
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

# =============================================================================
# Constants
# =============================================================================
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
NX, NY = 16, 12

FIG_W, FIG_H = 7.9, 5.3
FIG_DPI = 110

# Mapa
COLOR_FAIL_WEAK = "#E07070"      # vermelho fraco (errados)
COLOR_POS_XT_WEAK = "#2F80ED"    # azul fraco (ΔxT > 0)
COLOR_NONPOS_WEAK = "#A0A0A0"    # cinza fraco (ΔxT <= 0)
ALPHA_FAIL = 0.45
ALPHA_POS = 0.30
ALPHA_NONPOS = 0.22

# Bonificação por pressão (conservadora)
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
# xT Grid (lógica do app de referência)
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
# Geração de 250 passes de lateral direito (plausíveis)
# =============================================================================
def clip_pitch(x, y):
    return float(np.clip(x, 0.0, FIELD_X)), float(np.clip(y, 0.0, FIELD_Y))

def choose_pressure_tag(rng):
    # distribuição plausível: mais 0-0 e 1-0; menos 1-1
    tags = ["0-0", "1-0", "0-1", "1-1"]
    probs = [0.42, 0.28, 0.20, 0.10]
    return rng.choice(tags, p=probs)

def generate_rb_passes(n=250, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # tipos de passe típicos de lateral direito
    # weights somam 1.0
    pass_types = [
        ("recycle_back", 0.26),      # recuo / segurança
        ("line_up_right", 0.24),     # progressão no corredor
        ("inside_link", 0.18),       # passe por dentro
        ("switch_diagonal", 0.10),   # virada longa
        ("final_third_cross", 0.12), # cruzamento/bola na área
        ("underlap_cutback", 0.10),  # infiltração e passe atrás
    ]
    names = [p[0] for p in pass_types]
    probs = [p[1] for p in pass_types]

    for i in range(n):
        ptype = rng.choice(names, p=probs)

        # origem típica de lateral direito (lado direito alto de y no statsbomb)
        if ptype == "recycle_back":
            x0 = rng.uniform(35, 80)
            y0 = rng.uniform(62, 79)
            dx = rng.uniform(-22, -6)
            dy = rng.uniform(-10, 6)

        elif ptype == "line_up_right":
            x0 = rng.uniform(25, 75)
            y0 = rng.uniform(60, 79)
            dx = rng.uniform(8, 24)
            dy = rng.uniform(-6, 8)

        elif ptype == "inside_link":
            x0 = rng.uniform(30, 85)
            y0 = rng.uniform(60, 79)
            dx = rng.uniform(5, 18)
            dy = rng.uniform(-22, -8)

        elif ptype == "switch_diagonal":
            x0 = rng.uniform(30, 80)
            y0 = rng.uniform(62, 79)
            dx = rng.uniform(25, 48)
            dy = rng.uniform(-38, -18)

        elif ptype == "final_third_cross":
            x0 = rng.uniform(78, 110)
            y0 = rng.uniform(66, 79)
            dx = rng.uniform(4, 14)
            dy = rng.uniform(-26, -10)

        else:  # underlap_cutback
            x0 = rng.uniform(72, 102)
            y0 = rng.uniform(58, 75)
            dx = rng.uniform(5, 16)
            dy = rng.uniform(-18, -6)

        x1, y1 = clip_pitch(x0 + dx, y0 + dy)

        pressure_tag = choose_pressure_tag(rng)

        # modelo simples de sucesso: cai com pressão e distância
        dist = float(np.hypot(x1 - x0, y1 - y0))
        base_success = 0.88
        pressure_penalty = {"0-0": 0.00, "1-0": 0.06, "0-1": 0.06, "1-1": 0.12}[pressure_tag]
        dist_penalty = 0.10 if dist > 30 else (0.05 if dist > 20 else 0.0)
        success_prob = np.clip(base_success - pressure_penalty - dist_penalty, 0.45, 0.95)
        is_won = bool(rng.random() < success_prob)

        rows.append(
            {
                "number": i + 1,
                "seta": f"Seta {i+1}",
                "x_start": round(x0, 2),
                "y_start": round(y0, 2),
                "x_end": round(x1, 2),
                "y_end": round(y1, 2),
                "pressure_tag": pressure_tag,
                "pressure_label": PRESSURE_LABEL[pressure_tag],
                "is_won": is_won,
                "type": "PASS WON" if is_won else "PASS LOST",
                "profile": ptype
            }
        )

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def build_dataset(seed=42, n=250):
    d = generate_rb_passes(n=n, seed=seed).copy()
    d["outcome"] = np.where(d["is_won"], "completed", "incomplete")
    d["xt_start"] = d.apply(lambda r: xt_value(r.x_start, r.y_start), axis=1)
    d["xt_end"] = d.apply(lambda r: xt_value(r.x_end, r.y_end), axis=1)
    d["delta_xt_raw"] = np.where(d["is_won"], d["xt_end"] - d["xt_start"], 0.0)
    d["pressure_bonus"] = d["pressure_tag"].map(PRESSURE_BONUS).fillna(0.0)
    d["delta_xt_final"] = np.where(
        d["is_won"],
        d["delta_xt_raw"] * (1 + d["pressure_bonus"]),
        0.0
    )
    d["pass_distance"] = np.sqrt((d.x_end - d.x_start)**2 + (d.y_end - d.y_start)**2)
    return d

# =============================================================================
# Draw + Stats
# =============================================================================
def _base_pitch():
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a1a2e",
        line_color="#ffffff",
        line_alpha=0.95
    )
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#FFD54F", lw=1.0, alpha=0.18)
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.6, alpha=0.10, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig):
    fig.patches.append(FancyArrowPatch(
        (0.45, 0.05), (0.55, 0.05),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color="#cccccc"
    ))
    fig.text(0.5, 0.02, "Attacking Direction",
             ha="center", va="center", fontsize=9, color="#cccccc")

def _save_fig(fig):
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)

def draw_pass_map(df_plot: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()

    for _, row in df_plot.iterrows():
        if not row["is_won"]:
            color = COLOR_FAIL_WEAK
            alpha = ALPHA_FAIL
        else:
            if float(row["delta_xt_final"]) > 0:
                color = COLOR_POS_XT_WEAK
                alpha = ALPHA_POS
            else:
                color = COLOR_NONPOS_WEAK
                alpha = ALPHA_NONPOS

        pitch.arrows(
            row.x_start, row.y_start, row.x_end, row.y_end,
            color=color, width=1.55, headwidth=2.25, headlength=2.25,
            ax=ax, zorder=3, alpha=alpha
        )
        pitch.scatter(
            row.x_start, row.y_start, s=42, marker="o", color=color,
            edgecolors="white", linewidths=0.75, ax=ax, zorder=6, alpha=alpha
        )

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)

    legend = ax.legend(
        handles=[
            Line2D([0],[0], color=COLOR_NONPOS_WEAK, lw=2.5, label="Completed (ΔxT ≤ 0)", alpha=0.80),
            Line2D([0],[0], color=COLOR_POS_XT_WEAK, lw=2.5, label="Completed (ΔxT > 0)", alpha=0.85),
            Line2D([0],[0], color=COLOR_FAIL_WEAK, lw=2.5, label="Incomplete", alpha=0.85),
        ],
        loc="upper left", bbox_to_anchor=(0.01,0.99), frameon=True,
        facecolor="#1a1a2e", edgecolor="#444466", fontsize="x-small",
        labelspacing=0.5, borderpad=0.5
    )
    for t in legend.get_texts():
        t.set_color("white")
    legend.get_frame().set_alpha(0.92)

    _attack_arrow(fig)
    return _save_fig(fig), ax, fig

def compute_stats(df_stats: pd.DataFrame):
    total = len(df_stats)
    completed = int(df_stats["is_won"].sum())
    acc = round(completed / max(total, 1) * 100, 2)

    succ = df_stats[df_stats["is_won"]]
    sum_xt = float(succ["delta_xt_final"].sum()) if not succ.empty else 0.0
    mean_xt = float(succ["delta_xt_final"].mean()) if not succ.empty else 0.0
    pos_mask = succ["delta_xt_final"] > 0
    pos_pct = round(float(pos_mask.sum()) / max(total, 1) * 100, 2)

    by_pressure = (
        df_stats.groupby("pressure_tag", dropna=False)
        .agg(
            total=("number", "count"),
            completed=("is_won", "sum"),
            xt_sum=("delta_xt_final", "sum"),
            xt_mean=("delta_xt_final", "mean"),
        )
        .reset_index()
    )
    by_pressure["accuracy_pct"] = np.where(
        by_pressure["total"] > 0,
        by_pressure["completed"] / by_pressure["total"] * 100,
        0
    )
    by_pressure["pressure_label"] = by_pressure["pressure_tag"].map(PRESSURE_LABEL)

    return {
        "total": total,
        "completed": completed,
        "accuracy": acc,
        "sum_xt": sum_xt,
        "mean_xt": mean_xt,
        "pos_pct": pos_pct,
        "by_pressure": by_pressure.sort_values("pressure_tag")
    }

# =============================================================================
# Data load + tabs
# =============================================================================
with st.sidebar:
    st.markdown("### Configuração")
    seed = st.number_input("Seed aleatória", min_value=1, max_value=999999, value=42, step=1)
    n_passes = st.number_input("Quantidade de passes", min_value=50, max_value=1000, value=250, step=10)

df = build_dataset(seed=int(seed), n=int(n_passes))

tab_mapa, tab_analise = st.tabs(["📋 Mapa", "📈 Análise"])

# =============================================================================
# TAB 1 - Mapa
# =============================================================================
with tab_mapa:
    col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

    with col_filters:
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)

        st.markdown("### ⚡ Pressão")
        pressure_filter = st.radio(
            "Filtro de pressão",
            ["Todos", "0-0", "1-0", "0-1", "1-1"],
            index=0,
            key="pressure_filter"
        )

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)

        st.markdown("### 🎯 Passe")
        pass_filter = st.radio(
            "Filtro de passe",
            ["Todos", "Completos", "Errados", "ΔxT > 0", "ΔxT ≤ 0"],
            index=0,
            key="pass_filter"
        )

        st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
        st.markdown("### 🧭 Perfil")
        profile_filter = st.multiselect(
            "Perfis de passe",
            options=sorted(df["profile"].unique().tolist()),
            default=sorted(df["profile"].unique().tolist()),
            key="profile_filter"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    df_base = df.copy()

    if pressure_filter != "Todos":
        df_base = df_base[df_base["pressure_tag"] == pressure_filter]

    if pass_filter == "Completos":
        df_base = df_base[df_base["is_won"]]
    elif pass_filter == "Errados":
        df_base = df_base[~df_base["is_won"]]
    elif pass_filter == "ΔxT > 0":
        df_base = df_base[(df_base["is_won"]) & (df_base["delta_xt_final"] > 0)]
    elif pass_filter == "ΔxT ≤ 0":
        df_base = df_base[(df_base["is_won"]) & (df_base["delta_xt_final"] <= 0)]

    if profile_filter:
        df_base = df_base[df_base["profile"].isin(profile_filter)]

    df_base = df_base.reset_index(drop=True)

    with col_field:
        st.caption("Clique no ponto de origem para inspecionar o passe.")

        img_obj, ax, fig = draw_pass_map(df_base, f"Pass Map — Lateral Direito ({len(df_base)} passes filtrados)")
        click = streamlit_image_coordinates(img_obj, width=780, key="pm_map")
        selected_pass = None

        if click is not None and len(df_base) > 0:
            rw, rh = img_obj.size
            px = click["x"] * (rw / click["width"])
            py = click["y"] * (rh / click["height"])
            fx, fy = ax.transData.inverted().transform((px, rh - py))
            df_sel = df_base.copy()
            df_sel["_dist"] = np.sqrt((df_sel.x_start - fx)**2 + (df_sel.y_start - fy)**2)
            cands = df_sel[df_sel["_dist"] < 5.0].sort_values("_dist")
            if not cands.empty:
                selected_pass = cands.iloc[0]
        plt.close(fig)

        st.divider()
        st.subheader("Selected Event")
        if selected_pass is None:
            st.info("Clique em um passe no mapa para ver detalhes.")
        else:
            status = "✅ Completo" if selected_pass["is_won"] else "❌ Errado"
            st.success(
                f"{selected_pass['seta']} — {status} | {selected_pass['pressure_tag']} ({selected_pass['pressure_label']})"
            )
            c1, c2 = st.columns(2)
            c1.write(f"**Origem:** ({selected_pass.x_start:.2f}, {selected_pass.y_start:.2f})")
            c2.write(f"**Destino:** ({selected_pass.x_end:.2f}, {selected_pass.y_end:.2f})")

            c3, c4 = st.columns(2)
            c3.metric("xT início", f"{selected_pass.xt_start:.4f}")
            c4.metric("xT fim", f"{selected_pass.xt_end:.4f}")

            c5, c6 = st.columns(2)
            c5.metric("ΔxT bruto", f"{selected_pass.delta_xt_raw:.4f}")
            c6.metric("Bônus pressão", f"{selected_pass.pressure_bonus:.2f}")

            c7, c8 = st.columns(2)
            c7.metric("ΔxT final", f"{selected_pass.delta_xt_final:.4f}")
            c8.metric("Distância", f"{selected_pass.pass_distance:.1f} m")

            st.caption(f"Perfil simulado: **{selected_pass['profile']}**")

        with st.expander("📊 Full Pass Data Table", expanded=False):
            cols = [
                "number","seta","type","profile","pressure_tag","pressure_label",
                "x_start","y_start","x_end","y_end",
                "xt_start","xt_end","delta_xt_raw",
                "pressure_bonus","delta_xt_final","pass_distance"
            ]
            st.dataframe(
                df_base[cols].style.format({
                    "x_start":"{:.2f}","y_start":"{:.2f}",
                    "x_end":"{:.2f}","y_end":"{:.2f}",
                    "xt_start":"{:.4f}","xt_end":"{:.4f}",
                    "delta_xt_raw":"{:.4f}",
                    "pressure_bonus":"{:.2f}",
                    "delta_xt_final":"{:.4f}",
                    "pass_distance":"{:.1f}"
                }),
                use_container_width=True,
                height=350
            )

    with col_stats:
        s = compute_stats(df_base)

        with st.expander("📋 General Statistics", expanded=True):
            st.markdown('<div class="stats-section-title">Overview</div>', unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1: small_metric("Total", f"{s['total']}")
            with r2: small_metric("Completos", f"{s['completed']}")
            with r3: small_metric("Acurácia", f"{s['accuracy']:.1f}%")

            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
            st.markdown('<div class="stats-section-title">xT (final)</div>', unsafe_allow_html=True)
            x1, x2, x3 = st.columns(3)
            with x1: small_metric("Σ ΔxT", f"{s['sum_xt']:.3f}")
            with x2: small_metric("Média ΔxT", f"{s['mean_xt']:.3f}")
            with x3: small_metric("% ΔxT > 0", f"{s['pos_pct']:.1f}%")

        with st.expander("🧩 Por Cenário de Pressão", expanded=True):
            for _, row in s["by_pressure"].iterrows():
                st.markdown(
                    f"<div class='stats-section-title'>{row['pressure_tag']} — {row['pressure_label']}</div>",
                    unsafe_allow_html=True
                )
                a, b, c = st.columns(3)
                with a: small_metric("Total", f"{int(row['total'])}")
                with b: small_metric("Acurácia", f"{row['accuracy_pct']:.1f}%")
                with c: small_metric("Σ ΔxT", f"{float(row['xt_sum']):.3f}")
                st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)

        st.caption(
            "Bônus conservador de pressão sobre ΔxT bruto: 0-0=0.00, 1-0=0.06, 0-1=0.06, 1-1=0.12"
        )
        st.caption(
            "Mapa: cinza fraco = ΔxT ≤ 0 · azul fraco = ΔxT > 0 · vermelho fraco = passe errado"
        )

# =============================================================================
# TAB 2 - Análise
# =============================================================================
with tab_analise:
    st.subheader("Análise — Top 20 passes por ΔxT final")

    top20 = (
        df[df["is_won"]]
        .sort_values("delta_xt_final", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top20["rank"] = np.arange(1, len(top20) + 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de passes", f"{len(df)}")
    c2.metric("Passes completos", f"{int(df['is_won'].sum())}")
    c3.metric("Σ ΔxT (Top20)", f"{top20['delta_xt_final'].sum():.3f}")
    c4.metric("Média ΔxT (Top20)", f"{top20['delta_xt_final'].mean() if len(top20)>0 else 0:.3f}")

    show_cols = [
        "rank","number","seta","type","profile","pressure_tag","pressure_label",
        "x_start","y_start","x_end","y_end",
        "xt_start","xt_end","delta_xt_raw","pressure_bonus","delta_xt_final","pass_distance"
    ]

    st.dataframe(
        top20[show_cols].style.format({
            "x_start":"{:.2f}","y_start":"{:.2f}",
            "x_end":"{:.2f}","y_end":"{:.2f}",
            "xt_start":"{:.4f}","xt_end":"{:.4f}",
            "delta_xt_raw":"{:.4f}",
            "pressure_bonus":"{:.2f}",
            "delta_xt_final":"{:.4f}",
            "pass_distance":"{:.1f}"
        }),
        use_container_width=True,
        height=520
    )

    st.markdown("#### Coordenadas dos Top 20")
    coords_txt = []
    for _, r in top20.iterrows():
        coords_txt.append(
            f"#{int(r['rank'])} | {r['seta']} | {r['pressure_tag']} | "
            f"({r['x_start']:.2f}, {r['y_start']:.2f}) -> ({r['x_end']:.2f}, {r['y_end']:.2f}) | "
            f"ΔxTfinal={r['delta_xt_final']:.4f}"
        )
    st.code("\n".join(coords_txt) if coords_txt else "Sem dados.", language="text")
