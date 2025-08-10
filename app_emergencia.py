import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
import io

# ==================== CONFIG BÁSICA ====================
st.set_page_config(
    page_title="PREDWEEM",
    layout="wide",
    menu_items={
        "About": "PREDWEEM – Predictive Weed Emergence Models (UNS · INTA)",
    },
)

# CSS fijo para evitar problemas de f-strings
CSS = """
<style>
  .main { background-color: #f8f9f4; }
  h1, h2, h3, h4 { color: #2e5e2d; }
  .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; padding: 0.6em 1.2em; font-size: 1em; }
  .card { background:#fff; border:1px solid #e0e3da; border-radius:14px; padding:14px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; color:#fff; }
  .pill-bajo { background:#2ecc71; }.pill-medio { background:#f0932b; }.pill-alto { background:#d33535; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ==================== CONSTANTES NEGOCIO ====================
UMBRAL_MIN_DEF = 1.20
UMBRAL_MAX_DEF = 2.77
UMBRAL_AJUSTABLE_DEF = 2.00
VALOR_MAX_EMEAC_MODEL = 8.05  # para normalizar EMEAC (ajústalo si cambia el modelo)

# ==================== MODELO ANN (OPTIMIZADO) ====================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW          # esperado: (hidden, input)
        self.bias_IW = bias_IW  # (hidden,)
        self.LW = LW          # (out, hidden) típicamente (1, hidden)
        self.bias_out = bias_out  # (out,) típicamente (1,)
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
        """Forward pass vectorizado (≈10–30× más rápido que loop). Devuelve EMERREL (0–1) diario."""
        Xn = self.normalize_input(X_real)              # (n, in)
        # z1 = IW.T @ x + bias  -> batch: Xn @ IW.T + bias
        z1 = Xn @ self.IW.T + self.bias_IW            # (n, hidden)
        a1 = self.tansig(z1)
        # z2 = LW @ a1 + bias  -> batch: a1 @ LW.T + bias
        z2 = a1 @ self.LW.T + self.bias_out           # (n, out)
        y = self.tansig(z2).reshape(-1)               # (n,)
        emerrel_0_1 = self.desnormalize_output(y)
        # Asegurar [0,1]
        return np.clip(emerrel_0_1, 0.0, 1.0)

# ==================== CARGA DE PESOS (CACHE) ====================
@st.cache_resource(show_spinner=False)
def load_model() -> PracticalANNModel | None:
    base = Path(".")
    try:
        IW = np.load(base / "IW.npy")         # esperado (hidden, input)
        bias_IW = np.load(base / "bias_IW.npy")  # (hidden,)
        LW = np.load(base / "LW.npy")          # (out, hidden)
        bias_out = np.load(base / "bias_out.npy")  # (out,)
        return PracticalANNModel(IW, bias_IW, LW, bias_out)
    except Exception as e:
        st.warning(f"No se pudieron cargar pesos/sesgos del modelo: {e}
Usaré una señal DEMO.")
        return None

# ==================== UTILIDADES ====================
REQ_COLS = ("Julian_days", "TMAX", "TMIN", "Prec")

@st.cache_data(show_spinner=False)
def coerce_validate(df: pd.DataFrame) -> pd.DataFrame:
    if not set(REQ_COLS).issubset(df.columns):
        raise ValueError("Columnas requeridas: Julian_days, TMAX, TMIN, Prec")
    out = df.loc[:, REQ_COLS].copy()
    for c in REQ_COLS:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna().astype({"Julian_days": int}).sort_values("Julian_days").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fechas_desde_juliano(anio: int, julian: pd.Series) -> pd.Series:
    base = datetime(anio, 1, 1)
    return pd.to_datetime([base + timedelta(days=int(j)-1) for j in julian])

@st.cache_data(show_spinner=False)
def regla_precipitacion(prec: np.ndarray, emerrel: np.ndarray, ventana: int = 8, umbral_mm: float = 5.0, enabled: bool = True):
    if not enabled:
        return emerrel
    s = pd.Series(prec).rolling(window=ventana, min_periods=1).sum().to_numpy()
    mask = s >= umbral_mm
    out = emerrel.copy()
    out[~mask] = 0.0
    return out

@st.cache_data(show_spinner=False)
def calcular_series(model: PracticalANNModel | None, X: np.ndarray, prec: np.ndarray, umbral_100: float, aplicar_regla: bool):
    if model is not None:
        emerrel = model.predict_batch(X)
    else:
        # DEMO vectorizada
        tavg = (X[:,1] + X[:,2]) / 2.0
        tnorm = np.clip((tavg - 0.0)/40.0, 0, 1)
        pr_eff = np.clip(prec/20.0, 0, 0.3)
        emerrel = np.clip(0.15 + 0.85*(0.6*tnorm + 0.4*pr_eff), 0, 1)

    emerrel = regla_precipitacion(prec, emerrel, enabled=aplicar_regla)

    # EMEAC acumulado como fracción (0-1) usando VALOR_MAX_EMEAC_MODEL
    emerrel_cumsum = np.cumsum(emerrel)
    emer_ac = emerrel_cumsum / VALOR_MAX_EMEAC_MODEL
    # Diferencial diario
    emerrel_diff = np.diff(emer_ac, prepend=0)

    # EMEAC (%) segun umbrales
    emeac_min = np.clip(emerrel_cumsum / UMBRAL_MIN_DEF, 0, 1) * 100.0
    emeac_max = np.clip(emerrel_cumsum / UMBRAL_MAX_DEF, 0, 1) * 100.0
    emeac_adj = np.clip(emerrel_cumsum / umbral_100, 0, 1) * 100.0

    return emerrel_diff, emeac_min, emeac_max, emeac_adj

# ==================== CLASIFICACIÓN NIVELES ====================
def clasificar_nivel(valor: float) -> str:
    if valor < 0.02: return "Bajo"
    if valor <= 0.079: return "Medio"
    return "Alto"

# ==================== GRÁFICOS ====================
def plot_emerrel(fechas, emerrel_dif, nombre):
    niveles = np.array([clasificar_nivel(v) for v in emerrel_dif])
    colores = pd.Series(niveles).map({"Bajo":"green","Medio":"orange","Alto":"red"}).values
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(fechas, emerrel_dif, color=colores)
    ax.set_title(f"Emergencia Relativa Diaria – {nombre}")
    ax.set_ylabel("EMERREL (0–1)")
    ax.set_ylim(0, max(0.2, emerrel_dif.max()*1.1))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    st.pyplot(fig, use_container_width=True)


def plot_emeac(fechas, emeac_min, emeac_max, emeac_adj, nombre, mostrar_refs=False):
    fig, ax = plt.subplots(figsize=(12,4))
    # Banda entre min y max
    ax.fill_between(fechas, emeac_min, emeac_max, color='lightgray', alpha=0.35, label='Banda min–max')
    # Líneas
    ax.plot(fechas, emeac_adj, linewidth=2.2, label='Umbral ajustable')
    ax.plot(fechas, emeac_min, linestyle='--', linewidth=1.2, label='Umbral mínimo')
    ax.plot(fechas, emeac_max, linestyle='--', linewidth=1.2, label='Umbral máximo')

    if mostrar_refs:
        for nivel, color in zip([25, 50, 75, 90], ['#999', 'green', 'orange', 'red']):
            ax.axhline(nivel, linestyle='--', linewidth=1.0, color=color)

    ax.set_title(f"Progreso EMEAC (%) – {nombre}")
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Leyenda limpia
    ax.legend(loc='lower right')
    st.pyplot(fig, use_container_width=True)

# ==================== UI ====================
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=UMBRAL_MIN_DEF, max_value=UMBRAL_MAX_DEF, value=UMBRAL_MAX_DEF, step=0.01, format="%.2f"
)
mostrar_refs = st.sidebar.toggle("Mostrar líneas de referencia 25/50/75/90%", value=False)
aplicar_regla = st.sidebar.toggle("Aplicar regla de precipitación (n−7..n ≥ 5 mm)", value=True)

fecha_inicio = st.sidebar.date_input("Fecha inicio visualización", value=pd.to_datetime("2025-01-15")).to_pydatetime()
fecha_fin = st.sidebar.date_input("Fecha fin visualización", value=pd.to_datetime("2025-09-01")).to_pydatetime()

uploaded_files = st.file_uploader(
    "Suba uno o más .xlsx con columnas: Julian_days, TMAX, TMIN, Prec",
    type=["xlsx"], accept_multiple_files=True
)

# Modelo (cacheado)
model = load_model()

if not uploaded_files:
    st.info("Suba al menos un archivo .xlsx para iniciar el análisis.")
    st.stop()

for file in uploaded_files:
    try:
        raw = pd.read_excel(file)
        df = coerce_validate(raw)
    except Exception as e:
        st.warning(f"{file.name}: {e}")
        continue

    # Datos base
    X_real = df[["Julian_days","TMAX","TMIN","Prec"]].to_numpy()
    fechas = fechas_desde_juliano(2025, df["Julian_days"])  # año fijo 2025 (puede pasarse a sidebar)

    # Cálculo principal (cacheable)
    emerrel_diff, emeac_min, emeac_max, emeac_adj = calcular_series(
        model, X_real, df["Prec"].to_numpy(), umbral_usuario, aplicar_regla
    )

    # Filtrar por rango visual
    mask = (fechas >= np.datetime64(fecha_inicio)) & (fechas <= np.datetime64(fecha_fin))
    f, erd = fechas[mask], emerrel_diff[mask]
    emin, emax, eadj = emeac_min[mask], emeac_max[mask], emeac_adj[mask]

    nombre = Path(file.name).stem
    st.subheader(f"EMERREL (0–1) – {nombre}")
    plot_emerrel(f, erd, nombre)

    st.subheader(f"EMEAC (%) – {nombre}")
    plot_emeac(f, emin, emax, eadj, nombre, mostrar_refs)

    # Tabla de resultados y descargas
    niveles = np.array(["Bajo" if v<0.02 else ("Medio" if v<=0.079 else "Alto") for v in emerrel_diff])
    tabla = pd.DataFrame({
        "Fecha": fechas,
        "Nivel_Emergencia_relativa": niveles,
        "EMEAC (%) - ajustable": np.round(emeac_adj, 1),
    })
    st.subheader(f"Datos calculados – {nombre}")
    st.dataframe(tabla.loc[mask, ["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]], use_container_width=True)

    csv = tabla[["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]].to_csv(index=False).encode("utf-8")
    st.download_button(f"Descargar CSV – {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")

st.caption("Desarrollado por: Departamento de Agronomía (UNS) & INTA Bordenave")
