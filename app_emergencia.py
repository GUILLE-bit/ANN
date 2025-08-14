# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

st.set_page_config(page_title="Predicción de Emergencia Agrícola con ANN", layout="wide")

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        # Orden esperado por este script: [Julian_days, TMAX, TMIN, Prec]
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

# =================== Config de fuentes (CSV público) ===================
CSV_URL_PAGES = "https://GUILLE-bit.github.io/ANN/meteo_daily.csv"
CSV_URL_RAW   = "https://raw.githubusercontent.com/GUILLE-bit/ANN/gh-pages/meteo_daily.csv"

@st.cache_data(ttl=900)  # 15 min
def load_public_csv():
    last_err = None
    for url in (CSV_URL_PAGES, CSV_URL_RAW):
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
            faltan = req - set(df.columns)
            if faltan:
                raise ValueError(f"Faltan columnas en CSV público: {', '.join(sorted(faltan))}")
            df = df.sort_values("Fecha").reset_index(drop=True)
            return df, url
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No pude leer el CSV ni desde Pages ni desde Raw. Último error: {last_err}")

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
    min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f"
)

# Botón para forzar recarga de datos cacheados
if st.sidebar.button("Forzar recarga de datos"):
    st.cache_data.clear()

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
    try:
        df_auto, url_usada = load_public_csv()
        dfs.append(("MeteoBahia_CSV", df_auto))
        st.caption(f"Fuente CSV primaria: {CSV_URL_PAGES}")
        st.caption(f"Fuente CSV alternativa (fallback): {CSV_URL_RAW}")
        st.success(f"CSV cargado desde: {url_usada} · Rango: {df_auto['Fecha'].min().date()} → {df_auto['Fecha'].max().date()} · {len(df_auto)} días")
    except Exception as e:
        st.error(f"No se pudo leer el CSV público (Pages ni Raw). Detalle: {e}")
else:
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

        # (Opcional) aviso fuera de rango desactivado
        # if detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
        #     st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento ({modelo.input_min} a {modelo.input_max}).")

        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

        # Umbrales y % EMEAC (acumulado anual)
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 3.0
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
        pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
        pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

        # --- Rango 1/feb → 1/sep (reinicio) ---
        years = pred["Fecha"].dt.year.unique()
        if len(years) == 1:
            yr = int(years[0])
        else:
            yr = int(st.sidebar.selectbox("Año a mostrar (reinicio 1/feb → 1/sep)", sorted(years)))

        fecha_inicio_rango = pd.Timestamp(year=yr, month=2, day=1)
        fecha_fin_rango    = pd.Timestamp(year=yr, month=9, day=1)

        mask = (pred["Fecha"] >= fecha_inicio_rango) & (pred["Fecha"] <= fecha_fin_rango)
        pred_vis = pred.loc[mask].copy()

        if pred_vis.empty:
            st.warning(f"No hay datos entre {fecha_inicio_rango.date()} y {fecha_fin_rango.date()} para {nombre}.")
            continue

        # Recalcular acumulados y % EMEAC dentro del rango (reiniciados)
        pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
        pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 1.2
        pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 3.0
        pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / umbral_usuario
        pred_vis["EMEAC (%) - mínimo (rango)"]      = pred_vis["EMEAC (0-1) - mínimo (rango)"] * 100
        pred_vis["EMEAC (%) - máximo (rango)"]      = pred_vis["EMEAC (0-1) - máximo (rango)"] * 100
        pred_vis["EMEAC (%) - ajustable (rango)"]   = pred_vis["EMEAC (0-1) - ajustable (rango)"] * 100

        # Media móvil dentro del rango
        pred_vis["EMERREL_MA5_rango"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
        colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

        # --------- Gráfico EMERREL (rango) ---------
        st.subheader("EMERGENCIA RELATIVA DIARIA")
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred_vis["Fecha"], pred_vis["EMERREL(0-1)"], color=colores_vis)
        ax_er.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5_rango"], linewidth=2.2, label="Media móvil 5 días (rango)")
        ax_er.legend(loc="upper right")
        ax_er.set_title("EMERGENCIA RELATIVA DIARIA")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlim(fecha_inicio_rango, fecha_fin_rango)
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_er.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_er)

        # --------- Gráfico EMEAC (rango) ---------
        st.subheader("EMERGENCIA ACUMULADA DIARIA")
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.fill_between(pred_vis["Fecha"],
                        pred_vis["EMEAC (%) - mínimo (rango)"],
                        pred_vis["EMEAC (%) - máximo (rango)"],
                        alpha=0.4, label="Rango entre mínimo y máximo (reiniciado)")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - ajustable (rango)"], linewidth=2.5,
                label="Umbral ajustable (reiniciado)")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - mínimo (rango)"], linestyle='--', linewidth=1.5,
                label="Umbral mínimo (reiniciado)")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - máximo (rango)"], linestyle='--', linewidth=1.5,
                label="Umbral máximo (reiniciado)")

        # Líneas horizontales + leyenda sin duplicados
        niveles = [25, 50, 75, 90]
        for nivel in niveles:
            ax.axhline(nivel, linestyle='--', linewidth=1.2, label=f'{nivel}%')

        ax.set_title("EMERGENCIA ACUMULADA DIARIA")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("EMEAC (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(fecha_inicio_rango, fecha_fin_rango)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        handles, labels = ax.get_legend_handles_labels()
        seen = set(); handles_dedup, labels_dedup = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                handles_dedup.append(h); labels_dedup.append(l); seen.add(l)
        ax.legend(handles_dedup, labels_dedup, title="Referencias", loc="lower right")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig)

        # --------- Tabla y descarga (Fecha, Julian_days, EMEAC (%) y Nivel de EMERREL) ---------
        st.subheader(f"Resultados (1/feb → 1/sep) - {nombre}")
        col_emeac = "EMEAC (%) - ajustable (rango)" if "EMEAC (%) - ajustable (rango)" in pred_vis.columns else "EMEAC (%) - ajustable"
        tabla = pred_vis[["Fecha", "Julian_days", "Nivel_Emergencia_relativa", col_emeac]].rename(
            columns={
                "Nivel_Emergencia_relativa": "Nivel de EMERREL",
                col_emeac: "EMEAC (%)"
            }
        )
        st.dataframe(tabla, use_container_width=True)
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar resultados (rango) - {nombre}", csv, f"{nombre}_resultados_rango.csv", "text/csv")

