import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

st.set_page_config(layout="wide", page_title="Action Map v2 - xT + Pressão")

st.title("Action Map v2 - xT + Pressão (somente passes informados)")

# =========================
# Configuração de campo/xT
# =========================
FIELD_X, FIELD_Y = 120.0, 80.0
NX, NY = 16, 12

@st.cache_data(show_spinner=False)
def compute_xt_grid(nx=16, ny=12):
    # Grade simples e estável para xT base (0..1),
    # favorecendo progressão para frente e corredor central.
    x = np.linspace(0, FIELD_X, nx)
    y = np.linspace(0, FIELD_Y, ny)
    X, Y = np.meshgrid(x, y)

    xp = X / FIELD_X
    yc = 1.0 - np.abs((Y / FIELD_Y) - 0.5) * 2.0  # centralidade

    xt = 0.80 * xp + 0.20 * yc
    xt = (xt - xt.min()) / (xt.max() - xt.min() + 1e-12)
    return xt

XT_GRID = compute_xt_grid(NX, NY)

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * (NX - 1), 0, NX - 1))
    iy = int(np.clip((y / FIELD_Y) * (NY - 1), 0, NY - 1))
    return float(XT_GRID[iy, ix])

# =========================
# Dados (somente seus passes)
# =========================
# status: "successful" / "failed"
# pressure_code: "0-0", "1-0", "0-1", "1-1"
raw_passes = [
    # 1-1
    ("Seta 1", 27.09, 76.41, 56.84, 66.76, "successful", "1-1"),
    ("Seta 1", 13.95, 77.07, 36.89, 76.57, "successful", "1-1"),
    ("Seta 4", 36.23, 77.24, 40.38, 78.73, "successful", "1-1"),
    ("Seta 10", 25.26, 77.07, 39.05, 75.74, "failed", "1-1"),  # PASSE ERRADO

    # 0-0
    ("Seta 3", 28.42, 76.24, 13.29, 64.27, "successful", "0-0"),
    ("Seta 5", 38.22, 54.30, 13.45, 34.85, "successful", "0-0"),
    ("Seta 6", 41.55, 55.63, 31.74, 34.35, "successful", "0-0"),
    ("Seta 7", 53.35, 61.61, 71.97, 75.91, "successful", "0-0"),
    ("Seta 8", 56.84, 62.77, 48.20, 38.67, "successful", "0-0"),

    # 1-0
    ("Seta 2", 31.91, 77.40, 16.78, 64.27, "successful", "1-0"),
    ("Seta 9", 78.45, 73.25, 80.45, 73.25, "successful", "1-0"),
    ("Seta 11", 23.93, 77.74, 33.90, 74.91, "failed", "1-0"),  # PASSE ERRADO

    # 0-1
    ("Seta 1", 21.10, 71.42, 46.70, 70.09, "successful", "0-1"),
]

df = pd.DataFrame(
    raw_passes,
    columns=["seta", "x_start", "y_start", "x_end", "y_end", "outcome", "pressure_code"]
)

PRESSURE_FACTOR = {
    "0-0": 1.00,  # sem pressão
    "1-0": 0.85,  # pressão na origem
    "0-1": 0.90,  # pressão no destino
    "1-1": 0.75,  # pressão em ambos
}

df["pressure_factor"] = df["pressure_code"].map(PRESSURE_FACTOR).astype(float)

# xT base
df["xt_start"] = df.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
df["xt_end"] = df.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)

# Delta xT só conta se passe foi certo
df["delta_xt"] = np.where(df["outcome"] == "successful", df["xt_end"] - df["xt_start"], 0.0)

# Ajuste por pressão
df["delta_xt_pressao"] = df["delta_xt"] * df["pressure_factor"]

# distância
df["distance_m"] = np.sqrt((df["x_end"] - df["x_start"])**2 + (df["y_end"] - df["y_start"])**2)

# =========================
# Filtros
# =========================
c1, c2 = st.columns([1,1])
with c1:
    pressure_filter = st.multiselect(
        "Filtrar por pressão",
        options=["0-0", "1-0", "0-1", "1-1"],
        default=["0-0", "1-0", "0-1", "1-1"]
    )
with c2:
    outcome_filter = st.multiselect(
        "Filtrar por resultado",
        options=["successful", "failed"],
        default=["successful", "failed"]
    )

df_plot = df[df["pressure_code"].isin(pressure_filter) & df["outcome"].isin(outcome_filter)].copy()

# =========================
# Métricas
# =========================
total = len(df_plot)
ok = int((df_plot["outcome"] == "successful").sum())
acc = (ok / total * 100) if total else 0.0
sum_xt = float(df_plot["delta_xt"].sum()) if total else 0.0
sum_xt_p = float(df_plot["delta_xt_pressao"].sum()) if total else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total de passes", total)
m2.metric("Passes certos", ok, f"{acc:.1f}%")
m3.metric("Σ ΔxT (base)", f"{sum_xt:.4f}")
m4.metric("Σ ΔxT (aj. pressão)", f"{sum_xt_p:.4f}", f"{(sum_xt_p - sum_xt):.4f}")

# =========================
# Mapa
# =========================
pitch = Pitch(pitch_type="statsbomb", pitch_color="#0f172a", line_color="#e5e7eb")
fig, ax = pitch.draw(figsize=(10, 7))
fig.set_facecolor("#0f172a")

# cor por pressão
pressure_color = {
    "0-0": "#22c55e",  # verde
    "1-0": "#f59e0b",  # laranja
    "0-1": "#3b82f6",  # azul
    "1-1": "#ef4444",  # vermelho
}

# largura por ganho ajustado
max_abs = max(0.001, float(np.abs(df_plot["delta_xt_pressao"]).max())) if not df_plot.empty else 0.001

for _, r in df_plot.iterrows():
    c = pressure_color[r["pressure_code"]]
    lw = 1.3 + 4.0 * (abs(r["delta_xt_pressao"]) / max_abs)

    # transparência menor se falhou
    alpha = 0.90 if r["outcome"] == "successful" else 0.35

    ax.annotate(
        "",
        xy=(r["x_end"], r["y_end"]),
        xytext=(r["x_start"], r["y_start"]),
        arrowprops=dict(arrowstyle="->", color=c, lw=lw, alpha=alpha),
        zorder=3
    )

    # marcador origem/destino
    ax.scatter(r["x_start"], r["y_start"], s=28, c=c, edgecolors="white", linewidths=0.6, alpha=alpha, zorder=4)
    ax.scatter(r["x_end"], r["y_end"], s=52, c=c, marker="D", edgecolors="white", linewidths=0.6, alpha=alpha, zorder=5)

ax.set_title("Passes com xT Ajustado por Pressão", color="white", fontsize=14, pad=10)

legend_items = [
    Line2D([0], [0], color="#22c55e", lw=3, label="0-0 sem pressão"),
    Line2D([0], [0], color="#f59e0b", lw=3, label="1-0 pressão origem"),
    Line2D([0], [0], color="#3b82f6", lw=3, label="0-1 pressão destino"),
    Line2D([0], [0], color="#ef4444", lw=3, label="1-1 pressão ambos"),
]
leg = ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color("white")

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# =========================
# Tabela final
# =========================
st.markdown("### Tabela de passes")
show_cols = [
    "seta", "pressure_code", "outcome",
    "x_start", "y_start", "x_end", "y_end",
    "xt_start", "xt_end", "delta_xt", "pressure_factor", "delta_xt_pressao", "distance_m"
]
st.dataframe(
    df_plot[show_cols].sort_values(["pressure_code", "seta"]).reset_index(drop=True),
    use_container_width=True
)

st.caption(
    "Regra aplicada: ΔxT_ajustado = ΔxT_base × fator_de_pressão, "
    "com fatores 0-0=1.00, 1-0=0.85, 0-1=0.90, 1-1=0.75; passe errado => ΔxT=0."
)
