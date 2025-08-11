# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

st.set_page_config(page_title="Predicción de Emergencia Agrícola con ANN", layout="wide")

# ------------------- Modelo ANN (tu código original) ---------------------
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalize_output(self, y_norm, ymin=-1, ymax=1):
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)

        def clasificar(valor):
            if valor < 0.02:
                return "Bajo"
            elif valor <= 0.079:
                return "Medio"
            else:
                return "Alto"

        riesgo = np.array([clasificar(v) for v in emerrel_diff])

        return pd.DataFrame({
            "EMERREL(0-1)": emerrel_diff,
            "Nivel_Emergencia_relativa": riesgo
        })

# ------------------ Utilidades de datos ------------------
API_URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"

@st.cache_data(ttl=900)  # cachea 15 min para no saturar la API
def fetch_meteobahia_xml(url=API_URL, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def parse_meteobahia_daily(xml_bytes: bytes) -> pd.DataFrame:
    """
    Estructura (según ejemplo):
    <weatherdata><forecast><tabular>
      <day> <fecha value="YYYY-MM-DD"/> <tmax value=".."/> <tmin/> <precip value=".."/> ...
    """
    root = ET.fromstring(xml_bytes)
    days = root.findall(".//forecast/tabular/day")
    rows = []
    for d in days:
        fecha = d.find("./fecha")
        tmax = d.find("./tmax")
        tmin = d.find("./tmin")
        precip = d.find("./precip")

        fecha_val = fecha.get("value") if fecha is not None else None
        tmax_val = tmax.get("value") if tmax is not None else None
        tmin_val = tmin.get("value") if tmin is not None else None
        precip_val = precip.get("value") if precip is not None else None

        if not fecha_val:
            continue

        def to_float(x):
            try:
                return float(str(x).replace(",", "."))
            except:
                return None

        rows.append({
            "Fecha": pd.to_datetime(fecha_val).normalize(),
            "TMAX": to_float(tmax_val),
            "TMIN": to_float(tmin_val),
            "Prec": to_float(precip_val) if precip_val is not None else 0.0,
        })

    if not rows:
        raise RuntimeError("No se encontraron días en el XML.")
    df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    # Reorden a formato requerido por el modelo
    return df[["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]]

def validar_columnas(df: pd.DataFrame) -> tuple[bool, str]:
    req = {"Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    if faltan:
        return False, f"Faltan columnas: {', '.join(sorted(faltan))}"
    return True, ""

def obtener_colores(niveles: pd.Series):
    m = niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})
    return m.fillna("gray")

def detectar_fuera_rango(X_real: np.ndarray, input_min: np.ndarray, input_max: np.ndarray) -> bool:
    out = (X_real < input_min) | (X_real > input_max)
    return bool(np.any(out))

# ------------------ UI ------------------
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Fuente de datos")
modo = st.sidebar.radio("Elegí la fuente", ["Automático (API MeteoBahía)", "Subir Excel"])

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=1.2, max_value=3.0, value=2.90, step=0.01, format="%.2f"
)

# Cargar pesos del modelo
base = Path(__file__).parent
try:
    IW = np.load(base / "IW.npy")
    bias_IW = np.load(base / "bias_IW.npy")
    LW = np.load(base / "LW.npy")
    bias_out = np.load(base / "bias_out.npy")
except FileNotFoundError as e:
    st.error(f"Error al cargar archivos del modelo: {e}")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# ------------------ Obtener DataFrames según fuente ------------------
dfs = []  # lista de tuplas (nombre, df)

if modo == "Automático (API MeteoBahía)":
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Actualizar ahora"):
            fetch_meteobahia_xml.clear()  # limpia cache
    try:
        xml_bytes = fetch_meteobahia_xml()
        df_api = parse_meteobahia_daily(xml_bytes)
        dfs.append(("MeteoBahia_API", df_api))
        st.success(f"Datos automáticos cargados: {df_api['Fecha'].min().date()} → {df_api['Fecha'].max().date()} · {len(df_api)} días")
    except Exception as e:
        st.error(f"No se pudo obtener datos desde la API: {e}")
else:
    uploaded_files = st.file_uploader(
        "Sube uno o más archivos Excel (.xlsx) con columnas: Julian_days, TMAX, TMIN, Prec",
        type=["xlsx"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            df_up = pd.read_excel(file)
            ok, msg = validar_columnas(df_up)
            if not ok:
                st.warning(f"{file.name}: {msg}")
                continue
            # Si no trae Fecha, la construimos desde el año en curso (opcional)
            if "Fecha" not in df_up.columns:
                # asume año actual; ajusta si necesitas otra referencia
                year = pd.Timestamp.now().year
                df_up["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df_up["Julian_days"] - 1, unit="D")
            dfs.append((Path(file.name).stem, df_up))
    else:
        st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")

# ------------------ Procesamiento y gráficos ------------------
if dfs:
    for nombre, df in dfs:
        # Validación + orden
        ok, msg = validar_columnas(df)
        if not ok:
            st.warning(f"{nombre}: {msg}")
            continue

        df = df.sort_values("Julian_days").reset_index(drop=True)

        # Entradas al modelo
        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(dtype=float)
        fechas = df["Fecha"] if "Fecha" in df.columns else (pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D"))

        # Aviso: valores fuera del rango de entrenamiento
        if detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
            st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento ({modelo.input_min} a {modelo.input_max}). Resultados pueden degradarse.")

        pred = modelo.predict(X_real)
        pred["Fecha"] = pd.to_datetime(fechas)
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

        # Umbrales
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 3.0
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
        pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
        pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

        colores = obtener_colores(pred["Nivel_Emergencia_relativa"])

        # Rango de fechas dinámico
        fecha_inicio = pred["Fecha"].min()
        fecha_fin = pred["Fecha"].max()

        # --------- Gráfico EMERREL ---------
        st.subheader(f"EMERREL (0-1) - {nombre}")
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred["Fecha"], pred["EMERREL(0-1)"], color=colores)
        ax_er.plot(pred["Fecha"], pred["EMERREL_MA5"], linewidth=2.2, label="Media móvil 5 días")
        ax_er.legend(loc="upper right")
        ax_er.set_title(f"Emergencia Relativa Diaria - {nombre}")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlim(fecha_inicio, fecha_fin)
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_er.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_er)

        # --------- Gráfico EMEAC ---------
        st.subheader(f"EMEAC (%) - {nombre}")
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.fill_between(pred["Fecha"], pred["EMEAC (%) - mínimo"], pred["EMEAC (%) - máximo"], alpha=0.4, label="Rango entre mínimo y máximo")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - ajustable"], linewidth=2.5, label="Umbral ajustable")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - mínimo"], linestyle='--', linewidth=1.5, label="Umbral mínimo")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - máximo"], linestyle='--', linewidth=1.5, label="Umbral máximo")

        # Líneas horizontales correctas + leyendas
        niveles = [25, 50, 75, 90]
        for nivel in niveles:
            ax.axhline(nivel, linestyle='--', linewidth=1.2, label=f'{nivel}%')

        ax.set_title(f"Progreso EMEAC (%) - {nombre}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("EMEAC (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(fecha_inicio, fecha_fin)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        handles, labels = ax.get_legend_handles_labels()
        # Quita duplicados de legend manteniendo orden
        seen = set()
        handles_dedup, labels_dedup = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                handles_dedup.append(h); labels_dedup.append(l); seen.add(l)
        ax.legend(handles_dedup, labels_dedup, title="Referencias", loc="lower right")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig)

        # --------- Tabla y descarga ---------
        st.subheader(f"Datos calculados - {nombre}")
        columnas = ["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%) - ajustable"]
        st.dataframe(pred[columnas], use_container_width=True)
        csv = pred[columnas].to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")

