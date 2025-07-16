
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

    def desnormalize_output(self, y_norm, ymin=-1, ymax=1, xmin=0, xmax=0.5096):
        return (y_norm - ymin) * (xmax - xmin) / (ymax - ymin) + xmin

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        emer_ac = emerrel_cumsum / 8.210732682
        emerrel_diff = np.diff(emer_ac, prepend=0)

        def clasificar(valor):
            if valor < 0.02:
                return "Bajo"
            elif valor < 0.04:
                return "Medio"
            else:
                return "Alto"

        riesgo = np.array([clasificar(v) for v in emerrel_diff])

        return pd.DataFrame({
            "EMER_AC_pred": emer_ac,
            "EMERREL(0-1)": emerrel_diff,
            "Riesgo_Emergencia": riesgo
        })

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

# Cargar pesos
base = r"D:\TRABAJO 2024\machine Learning\Amaranthus"
IW = np.load(os.path.join(base, "IW.npy"))
bias_IW = np.load(os.path.join(base, "bias_IW.npy"))
LW = np.load(os.path.join(base, "LW.npy"))
bias_out = np.load(os.path.join(base, "bias_out.npy"))

# Instanciar modelo
modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# Cargar datos reales
df = pd.read_excel(os.path.join(base, "2024.xlsx"))
X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()

# Ejecutar predicción
salidas = modelo.predict(X_real)

# Generar fechas desde 1 de enero de 2025
fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
salidas["Fecha"] = fechas

# Filtrar entre día 32 y 210
salidas_filtradas = salidas[(df["Julian_days"] >= 32) & (df["Julian_days"] <= 210)]

# === GRÁFICO EMER_AC_pred ===
#plt.figure(figsize=(10, 4))
#plt.plot(salidas_filtradas["Fecha"], salidas_filtradas["EMER_AC_pred"], label="Emergencia Acumulada", color="blue")
#plt.axhline(0.1, color="green", linestyle="--", label="Límite Bajo (0.1)")
#plt.axhline(0.5, color="orange", linestyle="--", label="Límite Medio (0.5)")
#plt.axhline(0.9, color="red", linestyle="--", label="Límite Alto (0.9)")
#plt.title("Emergencia Acumulada (2025)")
#plt.xlabel("Fecha")
#plt.ylabel("EMER_AC_pred (0-1)")
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()

# === GRÁFICO EMERREL(0-1) CON BANDAS DE RIESGO ===
plt.figure(figsize=(10, 4))
color_map = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
colors = salidas_filtradas["Riesgo_Emergencia"].map(color_map)

plt.bar(salidas_filtradas["Fecha"], salidas_filtradas["EMERREL(0-1)"], color=colors, label="EMERREL(0-1)")
plt.title("Emergencia Relativa Diaria (2025)")
plt.xlabel("Fecha")
plt.ylabel("EMERREL(0-1)")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
