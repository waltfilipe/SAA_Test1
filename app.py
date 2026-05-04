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
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Pass Map Dashboard — Pressão")

# ── Style ────────────────────────────────────────────────────────────────────
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


st.title("Pass Map Dashboard — Pressão")

# ── Constants ────────────────────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
FIG_W, FIG_H = 8.2, 5.3
FIG_DPI = 110

COLOR_COMPLETED = "#c8c8c8"
COLOR_INCOMPLETE = "#E07070"
COLOR_11 = "#EF4444"  # pressão origem e destino
COLOR_10 = "#F59E0B"  # pressão origem
COLOR_01 = "#8B5CF6"  # pressão destino
COLOR_00 = "#22C55E"  # sem pressão
ALPHA_COMPLETED = 0.85
ALPHA_INCOMPLETE = 0.90

PRESSURE_LABELS = {
    "1-1": "Pressão origem + destino",
    "1-0": "Pressão só na origem",
    "0-1": "Pressão só no destino",
    "0-0": "Sem pressão",
}

# ── Data (fornecida por você) ────────────────────────────────────────────────
# columns: number, type, x_start, y_start, x_end, y_end, pressure
passes_raw = [
    # 1-1
    (1,  "PASS WON", 27.09, 76.41, 56.84, 66.76, "1-1"),
    (2,  "PASS WON", 13.95, 77.07, 36.89, 76.57, "1-1"),
    (4,  "PASS WON", 36.23, 77.24, 40.38, 78.73, "1-1"),
    (10, "PASS LOST",25.26, 77.07, 39.05, 75.74, "1-1"),

    # 0-0
    (3,  "PASS WON", 28.42, 76.24, 13.29, 64.27, "0-0"),
    (5,  "PASS WON", 38.22, 54.30, 13.45, 34.85, "0-0"),
    (6,  "PASS WON", 41.55, 55.63, 31.74, 34.35, "0-0"),
    (7,  "PASS WON", 53.35, 61.61, 71.97, 75.91, "0-0"),
    (8,  "PASS WON", 56.84, 62.77, 48.20, 38.67, "0-0"),

    # 1-0
    (2,  "PASS WON", 31.91, 77.40, 16.78, 64.27, "1-0"),
    (9,  "PASS WON", 78.45, 73.25, 80.45, 73.25, "1-0"),
    (11, "PASS LOST",23.93, 77.74, 33.90, 74.91, "1-0"),

    # 0-1
    (1,  "PASS WON", 21.10, 71.42, 46.70, 70.09, "0-1"),
]

df = pd.DataFrame(
    passes_raw,
    columns=["number", "type", "x_start", "y_start", "x_end", "y_end", "pressure"]
)

df["is_won"] = df["type"].str.contains("WON", case=False)
df["outcome"] = np.where(df["is_won"], "completed", "incomplete")
df["pass_distance"] = np.sqrt((df.x_end - df.x_start) ** 2 + (df.y_end - df.y_start) ** 2)


def split_pressure(col: pd.Series):
    origin = col.str.split("-").str[0].astype(int)
    dest = col.str.split("-").str[1].astype(int)
    return origin, dest

df["pressure_origin"], df["pressure_dest"] = split_pressure(df["pressure"])


# ── Helpers ──────────────────────────────────────────────────────────────────
def _save_fig(fig) -> Image.Image:
    fig.tight_layout()
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf)


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
    return fig, ax, pitch


def pressure_color(p: str):
    if p == "1-1":
        return COLOR_11
    if p == "1-0":
        return COLOR_10
    if p == "0-1":
        return COLOR_01
    return COLOR_00


def draw_pass_map(df_plot: pd.DataFrame, title: str):
    fig, ax, pitch = _base_pitch()

    for _, row in df_plot.iterrows():
        color = pressure_color(row["pressure"])
        alpha = ALPHA_COMPLETED if row["is_won"] else ALPHA_INCOMPLETE

        pitch.arrows(
            row.x_start, row.y_start, row.x_end, row.y_end,
            color=color, width=1.6, headwidth=2.3, headlength=2.3,
            ax=ax, zorder=3, alpha=alpha
        )
        pitch.scatter(
            row.x_start, row.y_start,
            s=48, marker="o", color=color,
            edgecolors="white", linewidths=0.8,
            ax=ax, zorder=6, alpha=alpha
        )

    ax.set_title(title, fontsize=12, color="#ffffff", pad=8)

    leg = ax.legend(
        handles=[
            Line2D([0], [0], color=COLOR_11, lw=2.8, label="1-1: Pressão origem + destino"),
            Line2D([0], [0], color=COLOR_10, lw=2.8, label="1-0: Pressão só na origem"),
            Line2D([0], [0], color=COLOR_01, lw=2.8, label="0-1: Pressão só no destino"),
            Line2D([0], [0], color=COLOR_00, lw=2.8, label="0-0: Sem pressão"),
            Line2D([0], [0], color=COLOR_INCOMPLETE, lw=2.8, linestyle="--", label="Tracejado = Incompleto"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize="x-small",
        labelspacing=0.5,
        borderpad=0.5
    )
    for t in leg.get_texts():
        t.set_color("white")
    leg.get_frame().set_alpha(0.92)

    return _save_fig(fig), ax, fig


def compute_stats(df_in: pd.DataFrame) -> dict:
    total = len(df_in)
    completed = int(df_in["is_won"].sum())
    incomplete = total - completed

    out = {
        "total_passes": total,
        "completed_passes": completed,
        "incomplete_passes": incomplete,
        "accuracy_pct": round((completed / total) * 100, 2) if total else 0.0,
        "avg_distance": round(float(df_in["pass_distance"].mean()), 2) if total else 0.0,
    }

    for p in ["1-1", "1-0", "0-1", "0-0"]:
        sub = df_in[df_in["pressure"] == p]
        t = len(sub)
        c = int(sub["is_won"].sum())
        out[f"{p}_total"] = t
        out[f"{p}_completed"] = c
        out[f"{p}_accuracy"] = round((c / t) * 100, 2) if t else 0.0
        out[f"{p}_pct_of_total"] = round((t / total) * 100, 2) if total else 0.0

    # pressão na origem / destino (independente da combinação)
    ori_p = int((df_in["pressure_origin"] == 1).sum())
    dst_p = int((df_in["pressure_dest"] == 1).sum())
    out["origin_pressure_total"] = ori_p
    out["dest_pressure_total"] = dst_p
    out["origin_pressure_pct"] = round((ori_p / total) * 100, 2) if total else 0.0
    out["dest_pressure_pct"] = round((dst_p / total) * 100, 2) if total else 0.0

    return out


# ── Layout ───────────────────────────────────────────────────────────────────
st.caption("Clique no ponto de origem para inspecionar o passe.")

col_filters, col_field, col_stats = st.columns([0.9, 2, 1], gap="large")

with col_filters:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Filtro de Pressão")
    pressure_filter = st.radio(
        "Selecione",
        ["Todos", "1-1", "1-0", "0-1", "0-0"],
        index=0
    )

    st.markdown('<hr class="filter-divider">', unsafe_allow_html=True)
    st.markdown("### ✅ Resultado")
    outcome_filter = st.radio(
        "Status do passe",
        ["Todos", "Completos", "Incompletos"],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_field:
    df_base = df.copy()

    if pressure_filter != "Todos":
        df_base = df_base[df_base["pressure"] == pressure_filter]

    if outcome_filter == "Completos":
        df_base = df_base[df_base["is_won"]]
    elif outcome_filter == "Incompletos":
        df_base = df_base[~df_base["is_won"]]

    df_base = df_base.reset_index(drop=True)

    DW = 820
    img_obj, ax, fig = draw_pass_map(df_base, "Pass Map — Pressão")
    click = streamlit_image_coordinates(img_obj, width=DW, key="pm_map")
    st.image(img_obj, width=DW)

    selected_pass = None
    if click is not None and not df_base.empty:
        rw, rh = img_obj.size
        px = click["x"] * (rw / click["width"])
        py = click["y"] * (rh / click["height"])
        fx, fy = ax.transData.inverted().transform((px, rh - py))

        df_sel = df_base.copy()
        df_sel["_dist"] = np.sqrt((df_sel.x_start - fx) ** 2 + (df_sel.y_start - fy) ** 2)
        cands = df_sel[df_sel["_dist"] < 5.0].sort_values("_dist")
        if not cands.empty:
            selected_pass = cands.iloc[0]

    plt.close(fig)

    st.divider()
    st.subheader("Evento Selecionado")
    if selected_pass is None:
        st.info("Clique em um ponto de origem no mapa para ver os detalhes.")
    else:
        status = "✅ Completo" if selected_pass["is_won"] else "❌ Incompleto"
        st.success(
            f"Passe #{int(selected_pass['number'])} — {status} | "
            f"{selected_pass['pressure']} ({PRESSURE_LABELS[selected_pass['pressure']]})"
        )
        c1, c2 = st.columns(2)
        c1.write(f"**Origem:** ({selected_pass.x_start:.2f}, {selected_pass.y_start:.2f})")
        c2.write(f"**Destino:** ({selected_pass.x_end:.2f}, {selected_pass.y_end:.2f})")
        st.metric("Distância do Passe", f"{selected_pass.pass_distance:.1f} m")

    with st.expander("📊 Tabela Completa"):
        cols = [
            "number", "type", "outcome", "pressure",
            "pressure_origin", "pressure_dest",
            "x_start", "y_start", "x_end", "y_end", "pass_distance"
        ]
        st.dataframe(
            df_base[cols].style.format({
                "x_start": "{:.2f}", "y_start": "{:.2f}",
                "x_end": "{:.2f}", "y_end": "{:.2f}",
                "pass_distance": "{:.1f}"
            }),
            use_container_width=True,
            height=360
        )

with col_stats:
    s = compute_stats(df_base)

    with st.expander("📋 Estatísticas Gerais", expanded=True):
        st.markdown('<div class="stats-section-title">Visão Geral</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            small_metric("Total", f"{s['total_passes']}")
        with r2:
            small_metric("Completos", f"{s['completed_passes']}")
        with r3:
            small_metric("Precisão", f"{s['accuracy_pct']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        a1, a2 = st.columns(2)
        with a1:
            small_metric("Pressão na Origem", f"{s['origin_pressure_total']}",
                         delta=f"{s['origin_pressure_pct']:.1f}%")
        with a2:
            small_metric("Pressão no Destino", f"{s['dest_pressure_total']}",
                         delta=f"{s['dest_pressure_pct']:.1f}%")

        st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)
        small_metric("Distância Média", f"{s['avg_distance']:.1f} m")

    with st.expander("🔬 Estatísticas por Cenário de Pressão", expanded=True):
        for p in ["1-1", "1-0", "0-1", "0-0"]:
            st.markdown(f"<div class='stats-section-title'>{p} — {PRESSURE_LABELS[p]}</div>",
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                small_metric("Total", f"{s[f'{p}_total']}")
            with c2:
                small_metric("Completos", f"{s[f'{p}_completed']}")
            with c3:
                small_metric("Precisão", f"{s[f'{p}_accuracy']:.1f}%")
            small_metric("% do Total", f"{s[f'{p}_pct_of_total']:.1f}%")
            st.markdown("<hr style='margin:6px 0 8px 0;'>", unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Cores por cenário de pressão: "
        "1-1 🔴 | 1-0 🟠 | 0-1 🟣 | 0-0 🟢. "
        "Vermelho no status indica passe incompleto."
    )
