import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Modelo ANN ---
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
        emer_ac = (emerrel_cumsum / valor_max_emeac)
        valor_max_emeac = 8.21
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

# --- Interfaz Streamlit ---
st.title("Predicción de Emergencia - Modelo ANN")

uploaded_file = st.file_uploader("Cargar archivo Excel con datos (2025)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
        base = os.path.dirname(__file__)
        IW = np.load(os.path.join(base, "IW.npy"))
        bias_IW = np.load(os.path.join(base, "bias_IW.npy"))
        LW = np.load(os.path.join(base, "LW.npy"))
        bias_out = np.load(os.path.join(base, "bias_out.npy"))

        modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        salidas = modelo.predict(X_real)

        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
        salidas["Fecha"] = fechas
        salidas = pd.concat([df["Julian_days"], salidas], axis=1)

        salidas_filtradas = salidas[(salidas["Julian_days"] >= 32) & (salidas["Julian_days"] <= 240)]
        salidas_filtradas["EMEAC"] = salidas_filtradas["EMERREL(0-1)"].cumsum()/8.21
        valor_max_emeac = 8.21
        salidas_filtradas["EMEAC(%)"] = (salidas_filtradas["EMEAC"] * 100) * 3

        # --- Gráfico EMERREL ---
        st.subheader("Emergencia Relativa Diaria (EMERREL(0-1))")
        fig, ax = plt.subplots(figsize=(10, 4))
        color_map = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
        colors = salidas_filtradas["Nivel_Emergencia_relativa"].map(color_map)
        ax.bar(salidas_filtradas["Fecha"], salidas_filtradas["EMERREL(0-1)"], color=colors)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("EMERREL(0-1)")
        ax.set_title("Emergencia Relativa 2025")
        ax.grid(True, linestyle='--', alpha=0.5)
        legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in color_map.items()]
        ax.legend(handles=legend_labels, title="Niveles")
        st.pyplot(fig)

        # --- Gráfico EMEAC(%) ---
        st.subheader("Progreso porcentual acumulado de Emergencia (EMEAC%)")
        fig_emeac, ax_emeac = plt.subplots(figsize=(10, 4))
        ax_emeac.bar(salidas_filtradas["Fecha"], salidas_filtradas["EMEAC(%)"], color="skyblue")
        niveles = {"10%": ("gray", 10), "25%": ("green", 25), "50%": ("orange", 50), "75%": ("red", 75), "90%": ("purple", 90)}
        for label, (color, yval) in niveles.items():
            ax_emeac.axhline(yval, color=color, linestyle="--", linewidth=1.5, label=label)
        import matplotlib.dates as mdates
        ax_emeac.set_xlabel("Fecha")
        ax_emeac.set_ylabel("EMEAC (%)")
        ax_emeac.set_title("Acumulado porcentual de Emergencia - 2025")
        ax_emeac.grid(True, linestyle="--", alpha=0.5)
        ax_emeac.set_ylim(0, 100)
        ax_emeac.xaxis.set_major_locator(mdates.DayLocator(interval=30))
        ax_emeac.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        ax_emeac.legend(title="Niveles de EMEAC")
        st.pyplot(fig_emeac)

        # --- Descargar resultados ---
        st.subheader("Resultados")
        st.dataframe(salidas_filtradas)
        csv = salidas_filtradas.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar resultados en CSV", csv, "predicciones_emergencia_2025.csv", "text/csv")

    else:
        st.error("El archivo debe contener las columnas: Julian_days, TMAX, TMIN, Prec.")
