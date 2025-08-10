import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path
from io import BytesIO

# ================== Estilo global Matplotlib ==================
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

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

st.sidebar.header("Configuración de visualización")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=1.2,
    max_value=2.77,
    value=2.77,
    step=0.01,
    format="%.2f"
)
mostrar_rango_completo = st.sidebar.checkbox(
    "Mostrar todo el rango temporal", value=True,
    help="Si se desactiva, se usará el rango enero–septiembre 2025"
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

# --------- Paleta de colores para niveles de riesgo ----------
COLOR_BAJO = "#2e7d32"    # verde
COLOR_MEDIO = "#ef6c00"   # naranja
COLOR_ALTO = "#c62828"    # rojo
COLOR_RANGO = "#b0b0b0"   # gris claro
COLOR_MIN_MAX = "#333333" # negro suave
COLOR_AJUSTABLE = "#1f77b4"  # azul estándar Matplotlib

def obtener_colores(niveles: pd.Series):
    return niveles.map({"Bajo": COLOR_BAJO, "Medio": COLOR_MEDIO, "Alto": COLOR_ALTO})

# --------- Utilidades de dibujo ----------
def formatear_eje_tiempo(ax, fechas, full_range=True):
    if full_range:
        # Locator automático y formateador conciso
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for label in ax.get_xticklabels():
        label.set(rotation=0, ha="center")

def descargar_figura(fig, nombre_png: str):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(
        label=f"Descargar gráfico PNG - {nombre_png}",
        data=buf.getvalue(),
        file_name=f"{nombre_png}.png",
        mime="image/png"
    )

# Rango de visualización por defecto si el usuario no elige todo
fecha_inicio_def = pd.to_datetime("2025-01-15")
fecha_fin_def = pd.to_datetime("2025-09-01")

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file)

        cols_req = ["Julian_days", "TMAX", "TMIN", "Prec"]
        if not all(col in df.columns for col in cols_req):
            st.warning(f"{file.name} no tiene las columnas requeridas: {', '.join(cols_req)}.")
            continue

        # Preparar datos
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

        # Definir ventana temporal
        if mostrar_rango_completo:
            x_min, x_max = pred["Fecha"].min(), pred["Fecha"].max()
        else:
            x_min, x_max = fecha_inicio_def, fecha_fin_def

        nombre = Path(file.name).stem
        colores = obtener_colores(pred["Nivel_Emergencia_relativa"])

        # =================== Gráfico EMERREL (barras) ===================
        st.subheader(f"EMERREL (0-1) - {nombre}")
        fig_er, ax_er = plt.subplots(figsize=(14, 5))
        # ancho ~0.9 día para que se vean levemente separadas
        ax_er.bar(pred["Fecha"], pred["EMERREL(0-1)"],
                  color=colores, edgecolor="#111111", linewidth=0.4, width=0.9)
        ax_er.set_title(f"Emergencia Relativa Diaria - {nombre}")
        ax_er.set_xlabel("Fecha")
        ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.grid(True)
        ax_er.set_xlim(x_min, x_max)
        formatear_eje_tiempo(ax_er, pred["Fecha"], full_range=mostrar_rango_completo)
        fig_er.tight_layout()
        st.pyplot(fig_er)
        descargar_figura(fig_er, f"{nombre}_EMERREL")

        # =================== Gráfico EMEAC (%) ===================
        st.subheader(f"EMEAC (%) - {nombre}")
        fig, ax = plt.subplots(figsize=(14, 5))

        # Banda entre mínimo y máximo
        ax.fill_between(pred["Fecha"],
                        pred["EMEAC (%) - mínimo"],
                        pred["EMEAC (%) - máximo"],
                        color=COLOR_RANGO, alpha=0.35, label="Rango entre mínimo y máximo")

        # Curvas
        ax.plot(pred["Fecha"], pred["EMEAC (%) - ajustable"],
                color=COLOR_AJUSTABLE, linewidth=2.6, label="Umbral ajustable")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - mínimo"],
                linestyle='--', color=COLOR_MIN_MAX, linewidth=1.5, label="Umbral mínimo")
        ax.plot(pred["Fecha"], pred["EMEAC (%) - máximo"],
                linestyle='--', color=COLOR_MIN_MAX, linewidth=1.5, label="Umbral máximo")

        # Líneas de referencia horizontales
        niveles_ref = [25, 50, 75, 90]
        colores_ref = ["#9e9e9e", COLOR_BAJO, COLOR_MEDIO, COLOR_ALTO]
        for nivel, c in zip(niveles_ref, colores_ref):
            ax.axhline(nivel, linestyle=':', color=c, linewidth=1.4)

        # Títulos, ejes y límites
        ax.set_title(f"Progreso EMEAC (%) - {nombre}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("EMEAC (%)")
        # Limitar a 0–100, pero permitir un pequeño margen superior si se pasa
        ymax = max(100, np.nanmax(pred[["EMEAC (%) - máximo", "EMEAC (%) - ajustable"]].to_numpy()))
        ax.set_ylim(0, min(110, ymax * 1.05))
        ax.set_xlim(x_min, x_max)
        ax.grid(True)
        formatear_eje_tiempo(ax, pred["Fecha"], full_range=mostrar_rango_completo)

        # Leyenda compacta y clara
        custom_legend = [
            Line2D([0], [0], color=COLOR_RANGO, linewidth=10, alpha=0.35, label="Rango entre mínimo y máximo"),
            Line2D([0], [0], color=COLOR_AJUSTABLE, linewidth=2.6, label="Umbral ajustable"),
            Line2D([0], [0], color=COLOR_MIN_MAX, linestyle='--', linewidth=1.5, label="Umbral mínimo"),
            Line2D([0], [0], color=COLOR_MIN_MAX, linestyle='--', linewidth=1.5, label="Umbral máximo"),
            Line2D([0], [0], color="#9e9e9e", linestyle=':', linewidth=1.4, label="25%"),
            Line2D([0], [0], color=COLOR_BAJO, linestyle=':', linewidth=1.4, label="50%"),
            Line2D([0], [0], color=COLOR_MEDIO, linestyle=':', linewidth=1.4, label="75%"),
            Line2D([0], [0], color=COLOR_ALTO, linestyle=':', linewidth=1.4, label="90%")
        ]
        ax.legend(handles=custom_legend, title="Referencias", loc="lower right", frameon=True)
        fig.tight_layout()
        st.pyplot(fig)
        descargar_figura(fig, f"{nombre}_EMEAC")

        # =================== Tabla de resultados ===================
        st.subheader(f"Datos calculados - {nombre}")
        columnas = ["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%) - ajustable"]
        tabla = pred[columnas].copy()
        tabla = tabla.sort_values("Fecha")
        tabla["EMEAC (%) - ajustable"] = tabla["EMEAC (%) - ajustable"].round(2)

        def estilo_tabla(df):
            def color_riesgo(val):
                m = {
                    "Bajo": f"background-color: #e8f5e9; color: {COLOR_BAJO}",
                    "Medio": f"background-color: #fff3e0; color: {COLOR_MEDIO}",
                    "Alto": f"background-color: #ffebee; color: {COLOR_ALTO}",
                }
                return m.get(val, "")
            return (df.style
                    .applymap(color_riesgo, subset=["Nivel_Emergencia_relativa"])
                    .format({"EMEAC (%) - ajustable": "{:.2f}"}))

        st.dataframe(estilo_tabla(tabla), use_container_width=True)

        # Descargas
        csv = tabla.to_csv(index=False).encode("utf-8")
        st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")

        xls_buf = BytesIO()
        with pd.ExcelWriter(xls_buf, engine="xlsxwriter") as writer:
            tabla.to_excel(writer, index=False, sheet_name="EMEAC")
        st.download_button(f"Descargar Excel - {nombre}", xls_buf.getvalue(),
                           f"{nombre}_EMEAC.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")

