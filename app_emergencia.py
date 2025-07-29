import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ------------------- MODELO ANN ---------------------
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

# ------------------ APP STREAMLIT --------------------
st.title("Predicción de Emergencia Agrícola con ANN")

# Umbral personalizable desde el usuario
st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=0.1,
    max_value=10.0,
    value=2.7,
    step=0.01,
    format="%.2f",
    help="Este umbral define el 100% del EMEAC para cualquier año"
)

# Carga de archivos y pesos del modelo
uploaded_files = st.file_uploader("Sube archivos Excel por año", type=["xlsx"], accept_multiple_files=True)

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

if uploaded_files:
    fig, ax = plt.subplots(figsize=(10, 4))
    resultados = []

    for file in uploaded_files:
        df = pd.read_excel(file)
        if not all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"Archivo {file.name} inválido: columnas faltantes.")
            continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

        predicciones = modelo.predict(X_real)
        predicciones["Fecha"] = fechas
        predicciones["Julian_days"] = df["Julian_days"]
        predicciones["EMERREL acumulado"] = predicciones["EMERREL(0-1)"].cumsum()

        # Escalar EMEAC(%) según el umbral proporcionado por el usuario
        predicciones["EMEAC (0-1)"] = predicciones["EMERREL acumulado"] / umbral_usuario
        predicciones["EMEAC (%)"] = predicciones["EMEAC (0-1)"] * 100

        # Nombre del archivo sin extensión como identificador
        label = Path(file.name).stem
        ax.plot(predicciones["Fecha"], predicciones["EMEAC (%)"], label=label, marker='o')

        # Guardar para posible descarga
        predicciones["Año"] = label
        resultados.append(predicciones)

    # Finalizar gráfico
    ax.set_title(f"Comparación EMEAC (%) con umbral: {umbral_usuario}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("EMEAC (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Combinar resultados en un DataFrame
    df_total = pd.concat(resultados, ignore_index=True)
    csv = df_total.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar resultados CSV", csv, "EMEAC_comparado.csv", "text/csv")

else:
    st.info("Sube al menos un archivo .xlsx con columnas: Julian_days, TMAX, TMIN, Prec.")

