
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ------------------- Modelo ANN ---------------------
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

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

# ------------------ Interfaz Streamlit ------------------
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=0.1,
    max_value=10.0,
    value=2.7,
    step=0.01,
    format="%.2f",
    help="Define el valor máximo acumulado de EMERREL para escalar EMEAC(%)"
)

uploaded_files = st.file_uploader(
    "Sube uno o más archivos Excel (.xlsx) con las columnas: Julian_days, TMAX, TMIN, Prec",
    type=["xlsx"],
    accept_multiple_files=True
)

# Cargar modelo
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

# Colores por nivel de EMERREL
def obtener_colores(niveles):
    return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})

legend_labels = [
    plt.Line2D([0], [0], color='green', lw=4, label='Bajo'),
    plt.Line2D([0], [0], color='orange', lw=4, label='Medio'),
    plt.Line2D([0], [0], color='red', lw=4, label='Alto')
]

# Rango de fechas para visualización
fecha_inicio = pd.to_datetime("2025-02-01")
fecha_fin = pd.to_datetime("2025-09-01")

# Procesar archivos subidos
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file)
        if not all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"El archivo {file.name} no tiene las columnas necesarias.")
            continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMEAC (0-1)"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%)"] = pred["EMEAC (0-1)"] * 100

        nombre = Path(file.name).stem
        colores = obtener_colores(pred["Nivel_Emergencia_relativa"])

        # --- Gráfico EMERREL (0-1) ---
        st.subheader(f"Emergencia Relativa Diaria - {nombre}")
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred["Fecha"], pred["EMERREL(0-1)"], color=colores)
        ax_er.set_title(f"EMERREL (0-1) Diario - {nombre}")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlim(fecha_inicio, fecha_fin)
        ax_er.legend(handles=legend_labels, title="Nivel")
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_er.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_er)

        # --- Gráfico EMEAC (%) como área ---
        st.subheader(f"Progreso acumulado de EMEAC (%) - {nombre}")
        fechas_validas = pd.to_datetime(pred["Fecha"])
        emeac_pct = pd.to_numeric(pred["EMEAC (%)"], errors="coerce")
        validez = ~(fechas_validas.isna() | emeac_pct.isna())
        fechas_plot = fechas_validas[validez].to_numpy()
        emeac_plot = emeac_pct[validez].to_numpy()

        fig_eac, ax_eac = plt.subplots(figsize=(14, 5), dpi=150)
        ax_eac.fill_between(fechas_plot, emeac_plot, color="skyblue", alpha=0.5)
        ax_eac.plot(fechas_plot, emeac_plot, color="blue", linewidth=2)
        ax_eac.set_title(f"EMEAC (%) Acumulado - {nombre} (Umbral: {umbral_usuario})")
        ax_eac.set_xlabel("Fecha")
        ax_eac.set_ylabel("EMEAC (%)")
        ax_eac.set_ylim(0, 110)
        ax_eac.set_xlim(fecha_inicio, fecha_fin)
        ax_eac.grid(True, linestyle="--", alpha=0.5)
        ax_eac.xaxis.set_major_locator(mdates.MonthLocator())
        ax_eac.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_eac.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_eac)

        # Mostrar y exportar
        st.subheader(f"Datos procesados - {nombre}")
        st.dataframe(pred)
        csv = pred.to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")
else:
    st.info("Sube al menos un archivo .xlsx para visualizar los resultados.")
