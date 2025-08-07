
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

# ------------------ Interfaz Streamlit ------------------
st.title("Predicción de Emergencia Agrícola con ANN")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=1.2,
    max_value=2.77,
    value=2.77,
    step=0.01,
    format="%.2f"
)

uploaded_files = st.file_uploader(
    "Sube uno o más archivos Excel (.xlsx) con columnas: Julian_days, TMAX, TMIN, Prec",
    type=["xlsx"],
    accept_multiple_files=True
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

# Función para asignar color según nivel
def obtener_colores(niveles):
    return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})

# Rango de visualización
fecha_inicio = pd.to_datetime("2025-01-15")
fecha_fin = pd.to_datetime("2025-09-01")

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file)
        if not all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"{file.name} no tiene las columnas requeridas.")
            continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
        pred = modelo.predict(X_real)

        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

        # Umbrales
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 2.77
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
        pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
        pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100

        nombre = Path(file.name).stem
        colores = obtener_colores(pred["Nivel_Emergencia_relativa"])

        # Gráfico EMERREL
        st.subheader(f"EMERREL (0-1) - {nombre}")
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred["Fecha"], pred["EMERREL(0-1)"], color=colores)
        ax_er.set_title(f"Emergencia Relativa Diaria - {nombre}")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlim(fecha_inicio, fecha_fin)
        ax_er.xaxis.set_major_locator(mdates.MonthLocator())
        ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.setp(ax_er.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig_er)

        # Gráfico EMEAC con área y líneas
        st.subheader(f"EMEAC (%) - {nombre}")
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.fill_between(pred["Fecha"], pred["EMEAC (%) - mínimo"], pred["EMEAC (%) - máximo"], color="lightgray", alpha=0.4, label="Rango entre mínimo y máximo")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - ajustable"], color="blue", linewidth=2.5, label="Umbral ajustable")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - mínimo"], linestyle='--', color="black", linewidth=1.5, label="Umbral mínimo")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - máximo"], linestyle='--', color="black", linewidth=1.5, label="Umbral máximo")

        for nivel, color in zip([25, 50, 75, 90], ['gray', 'green', 'orange', 'red']):
            ax.axhline(nivel, linestyle='--', color=color, linewidth=1.5, label=f'90%')

        ax.set_title(f"Progreso EMEAC (%) - {nombre}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("EMEAC (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(fecha_inicio, fecha_fin)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.legend(title="Referencias", loc="lower right")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        st.pyplot(fig)

        # Tabla de resultados
        st.subheader(f"Datos calculados - {nombre}")
        columnas = ["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%) - mínimo", "EMEAC (%) - ajustable", "EMEAC (%) - máximo"]
        st.dataframe(pred[columnas])
        csv = pred[columnas].to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")

else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")
