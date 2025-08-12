# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

st.set_page_config(page_title="Predicción de Emergencia Agrícola con ANN", layout="wide")

# =================== Modelo ANN (tu lógica original) ===================
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

# =================== Carga de datos (CSV público o Excel) ===================

CSV_URL_PAGES = "https://GUILLE-bit.github.io/ANN/meteo_daily.csv"
CSV_URL_RAW   = "https://raw.githubusercontent.com/GUILLE-bit/ANN/gh-pages/meteo_daily.csv"

@st.cache_data(ttl=900)
def load_public_csv():
    import pandas as pd, urllib.error
    try:
        return pd.read_csv(CSV_URL_PAGES, parse_dates=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    except Exception as e_pages:
        # Fallback al raw si Pages no está (404) o tarda en propagarse
        return pd.read_csv(CSV_URL_RAW, parse_dates=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

@st.cache_data(ttl=900)  # 15 min
def load_public_csv():
    df = pd.read_csv(CSV_URL, parse_dates=["Fecha"])
    # Garantiza columnas y orden
    req = ["Fecha", "Julian_days", "TMAX", "TMIN", "Prec"]
    faltan = set(req) - set(df.columns)
    if faltan:
        raise ValueError(f"CSV público no tiene columnas requeridas: {', '.join(sorted(faltan))}")
    return df.sort_values("Julian_days").reset_index(drop=True)

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

# =================== UI ===================
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio("Elegí la fuente", ["Automático (CSV público)", "Subir Excel"])

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

# =================== Obtener DataFrames ===================
dfs = []  # lista de (nombre, df)

if fuente == "Automático (CSV público)":
    st.caption(f"Fuente CSV: {CSV_URL}")
    try:
        df_auto = load_public_csv()
        dfs.append(("MeteoBahia_CSV", df_auto))
        st.success(f"CSV público cargado: {df_auto['Fecha'].min().date()} → {df_auto['Fecha'].max().date()} · {len(df_auto)} días")
    except Exception as e:
        st.error(f"No se pudo leer el CSV público: {e}. Probá más tarde o usa 'Subir Excel'.")

else:  # Subir Excel
    uploaded_files = st.file_uploader(
        "Sube uno o más archivos Excel (.xlsx) con columnas: Julian_days, TMAX, TMIN, Prec",
        type=["xlsx"], accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            df_up = pd.read_excel(file)
            ok, msg = validar_columnas(df_up)
            if not ok:
                st.warning(f"{file.name}: {msg}")
                continue
            if "Fecha" not in df_up.columns:
                # reconstruye Fecha desde el año actual si no viene
                year = pd.Timestamp.now().year
                df_up["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df_up["Julian_days"] - 1, unit="D")
            dfs.append((Path(file.name).stem, df_up))
    else:
        st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")

# =================== Procesamiento y gráficos ===================
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
        fechas = pd.to_datetime(df["Fecha"])

        # Aviso: entradas fuera del rango de entrenamiento
        if detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
            st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento ({modelo.input_min} a {modelo.input_max}).")

        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

        # Umbrales y % EMEAC
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

        # Líneas horizontales + leyenda sin duplicados
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
