import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import io
import hashlib

# ===================== CONFIGURACIÓN BÁSICA =====================
st.set_page_config(
    page_title="PREDWEEM – Emergencia de Malezas",
    layout="wide",
    menu_items={"About": "PREDWEEM · UNS & INTA"},
)

# —— Estilos ligeros ——
CSS = """
<style>
  .main { background:#f8f9f4; }
  h1, h2, h3, h4 { color:#2e5e2d; }
  .stButton>button { background:#4CAF50; color:#fff; border-radius:10px; padding:0.5em 1.1em; }
  .card { background:#fff; border:1px solid #e0e6db; border-radius:14px; padding:14px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PRIMARY_MIN = 1.20
PRIMARY_MAX = 2.77
VALOR_MAX_EMEAC_MODEL = 8.05  # normalización interna del modelo

REQ_COLS = ("Julian_days", "TMAX", "TMIN", "Prec")

# ===================== MODELO ANN (vectorizado) =====================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW          # (in, hidden)
        self.bias_IW = bias_IW  # (hidden,)
        self.LW = LW          # (out, hidden)
        self.bias_out = bias_out  # (out,)
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real: np.ndarray) -> np.ndarray:
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalize_output(self, y_norm, ymin=-1, ymax=1):
        # map [-1,1] -> [0,1]
        return (y_norm - ymin) / (ymax - ymin)

    def predict_batch(self, X_real: np.ndarray) -> np.ndarray:
        """
        Forward pass vectorizado (≈10–30× más rápido que loop).
        Devuelve EMERREL (0–1) diario como diferencial de EMEAC normalizado.
        """
        Xn = self.normalize_input(X_real)            # (n, in)
        # z1 = IW^T @ x + b    -> batch: Xn @ IW + b
        z1 = Xn @ self.IW + self.bias_IW             # (n, hidden)
        a1 = self.tansig(z1)
        # z2 = LW @ a1 + b     -> batch: a1 @ LW.T + b
        z2 = a1 @ self.LW.T + self.bias_out          # (n, out)
        y = self.tansig(z2).reshape(-1)              # (n,)
        emerrel_desnorm = self.desnormalize_output(y)  # (0..1)
        # Convertimos a EMEAC normalizado y luego diferencial diario
        emeac_norm = np.cumsum(emerrel_desnorm) / VALOR_MAX_EMEAC_MODEL
        emerrel_diff = np.diff(emeac_norm, prepend=0)
        return np.clip(emerrel_diff, 0, 1)

# ===================== CARGA DEL MODELO (cache) =====================
@st.cache_resource(show_spinner=False)
def load_model():
    base = Path(".")
    try:
        IW = np.load(base / "IW.npy")
        bias_IW = np.load(base / "bias_IW.npy")
        LW = np.load(base / "LW.npy")
        bias_out = np.load(base / "bias_out.npy")
        # Asegurar shapes consistentes
        # Esperamos IW shape (in, hidden); si viene (hidden, in), transponemos
        if IW.shape[0] != 4:
            IW = IW.T
        return PracticalANNModel(IW, bias_IW, LW, bias_out)
    except Exception as e:
        st.error(f"No pude cargar los pesos del modelo: {e}")
        return None

# ===================== UTILIDADES (cache) =====================
def _file_key(uploaded_file) -> str:
    b = uploaded_file.getbuffer()
    h = hashlib.md5(b).hexdigest()
    return f"{uploaded_file.name}:{h}"

@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

@st.cache_data(show_spinner=False)
def coerce_validate(df: pd.DataFrame) -> pd.DataFrame:
    if not set(REQ_COLS).issubset(df.columns):
        raise ValueError("Columnas requeridas: Julian_days, TMAX, TMIN, Prec")
    out = df.loc[:, REQ_COLS].copy()
    for c in REQ_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().astype({"Julian_days": int}).sort_values("Julian_days").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fechas_desde_juliano(anio: int, julian: pd.Series) -> pd.Series:
    base = datetime(anio, 1, 1)
    return pd.to_datetime([base + timedelta(days=int(j) - 1) for j in julian])

@st.cache_data(show_spinner=False)
def calcular_series(X: np.ndarray, umbral_100: float, model: PracticalANNModel) -> dict:
    emerrel = model.predict_batch(X)  # diferencial diario (0–1)
    emerrel_cumsum = np.cumsum(emerrel)  # acumulado (0.. VALOR_MAX_EMEAC_MODEL/VALOR_MAX_EMEAC_MODEL = ~1)
    emeac_min = np.clip(emerrel_cumsum / PRIMARY_MIN, 0, 1) * 100.0
    emeac_max = np.clip(emerrel_cumsum / PRIMARY_MAX, 0, 1) * 100.0
    emeac_adj = np.clip(emerrel_cumsum / umbral_100, 0, 1) * 100.0
    # Clasificación rápida
    niveles = np.where(emerrel < 0.02, "Bajo", np.where(emerrel <= 0.079, "Medio", "Alto"))
    return {
        "emerrel": emerrel,
        "emeac_min": emeac_min,
        "emeac_max": emeac_max,
        "emeac_adj": emeac_adj,
        "niveles": niveles,
    }

# ===================== UI =====================
st.title("🌱 PREDWEEM – Predicción de Emergencia con ANN")

with st.sidebar:
    st.header("Configuración")
    umbral_usuario = st.number_input(
        "Umbral de EMEAC para 100%",
        min_value=PRIMARY_MIN, max_value=PRIMARY_MAX,
        value=PRIMARY_MAX, step=0.01, format="%.2f"
    )
    anio_base = st.number_input("Año base (fechas calendario)", min_value=1900, max_value=2100, value=2025, step=1)
    mostrar_refs = st.toggle("Líneas de referencia 25/50/75/90%", value=False)
    st.divider()
    uploaded_files = st.file_uploader(
        "Suba uno o más .xlsx con columnas: Julian_days, TMAX, TMIN, Prec",
        type=["xlsx"], accept_multiple_files=True
    )
    with st.expander("Ayuda / Formato"):
        st.markdown(
            "- **Julian_days**: 1..365 (o 366)\n"
            "- **TMAX/TMIN**: °C\n"
            "- **Prec**: mm\n"
            "- Use la plantilla del portal para evitar errores de nombres."
        )

model = load_model()
if model is None:
    st.stop()

if not uploaded_files:
    st.info("👈 Cargue al menos un archivo .xlsx para comenzar.")
    st.stop()

# Paleta para barras por nivel
COLOR_MAP = {"Bajo": "green", "Medio": "orange", "Alto": "red"}

for uf in uploaded_files:
    key = _file_key(uf)
    try:
        df_raw = read_excel_cached(uf.getbuffer().tobytes())
        df = coerce_validate(df_raw)
    except Exception as e:
        st.warning(f"{uf.name}: {e}")
        continue

    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
    fechas = fechas_desde_juliano(anio_base, df["Julian_days"])
    series = calcular_series(X, umbral_usuario, model)

    # Rango sugerido para slider de fechas
    fecha_min, fecha_max = fechas.min(), fechas.max()
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader(Path(uf.name).stem)
    with c2:
        rango = st.slider(
            "Rango de fechas a visualizar",
            min_value=fecha_min.to_pydatetime(),
            max_value=fecha_max.to_pydatetime(),
            value=(max(fecha_min.to_pydatetime(), datetime(anio_base,1,15)),
                   min(fecha_max.to_pydatetime(), datetime(anio_base,9,1))),
            key=f"slider_{key}",
        )

    mask = (fechas >= rango[0]) & (fechas <= rango[1])
    f = fechas[mask]
    er = series["emerrel"][mask]
    emin = series["emeac_min"][mask]
    emax = series["emeac_max"][mask]
    eadj = series["emeac_adj"][mask]
    niveles = series["niveles"][mask]

    tabs = st.tabs(["📈 Gráficos", "🧾 Tabla", "⬇️ Descargas"])

    # --------- Tab Gráficos ---------
    with tabs[0]:
        st.markdown("#### EMERREL (0–1)")
        fig_er, ax_er = plt.subplots(figsize=(11, 3.8), dpi=150)
        colores = pd.Series(niveles).map(COLOR_MAP).values
        ax_er.bar(f, er, color=colores, width=1.0, align="center")
        ax_er.set_ylabel("EMERREL")
        ax_er.set_ylim(0, max(0.2, er.max()*1.1 if len(er) else 1))
        ax_er.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        st.pyplot(fig_er, use_container_width=True)

        st.markdown("#### EMEAC (%)")
        fig_ac, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
        ax.fill_between(f, emin, emax, color="lightgray", alpha=0.35, label="Banda min–max")
        ax.plot(f, eadj, linewidth=2.2, label="Umbral ajustable")
        ax.plot(f, emin, linestyle="--", linewidth=1.2, label="Umbral mínimo")
        ax.plot(f, emax, linestyle="--", linewidth=1.2, label="Umbral máximo")
        if mostrar_refs:
            for nivel, color in zip([25, 50, 75, 90], ["#999", "green", "orange", "red"]):
                ax.axhline(nivel, linestyle="--", linewidth=1.0, color=color)
        ax.set_ylabel("EMEAC (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="lower right")
        st.pyplot(fig_ac, use_container_width=True)

    # --------- Tab Tabla ---------
    with tabs[1]:
        tabla = pd.DataFrame({
            "Fecha": fechas,
            "Nivel_Emergencia_relativa": series["niveles"],
            "EMEAC (%) - ajustable": np.round(series["emeac_adj"], 1),
        })
        st.dataframe(tabla.loc[mask], use_container_width=True, height=360)

    # --------- Tab Descargas ---------
    with tabs[2]:
        # CSV ligero
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV",
            csv,
            file_name=f"{Path(uf.name).stem}_resultados.csv",
            mime="text/csv",
            use_container_width=True,
        )
        # Excel con hojas separadas
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            tabla.to_excel(writer, index=False, sheet_name="resultados")
        st.download_button(
            "Descargar Excel",
            buf.getvalue(),
            file_name=f"{Path(uf.name).stem}_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
