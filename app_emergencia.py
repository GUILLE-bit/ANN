
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from io import BytesIO
from pathlib import Path

# ===================== Apariencia global =====================
plt.rcParams.update({
    "figure.dpi": 170,
    "savefig.dpi": 170,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ===================== Modelo ANN (igual lógica) =====================
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

# ===================== Utilidades UI/Export =====================
def _fig_to_png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

def _excel_export(df: pd.DataFrame, filename: str) -> bytes:
    buf = BytesIO()
    try:
        import xlsxwriter  # si existe, usamos estilos
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            sheet = "resultados"
            df.to_excel(writer, index=False, sheet_name=sheet)
            wb = writer.book; ws = writer.sheets[sheet]
            fmt_date = wb.add_format({"num_format": "yyyy-mm-dd"})
            fmt_pct  = wb.add_format({"num_format": "0.0"})
            ws.set_column("A:A", 12, fmt_date)   # Fecha
            ws.set_column("B:B", 18)             # Nivel
            ws.set_column("C:C", 12, fmt_pct)    # EMEAC %
    except ModuleNotFoundError:
        # Fallback sin estilos
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="resultados")
    return buf.getvalue()

def _colores_por_nivel(series_nivel: pd.Series):
    return series_nivel.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})

# ===================== App =====================
st.set_page_config(page_title="PREDWEEM – UI Pro", layout="wide")
st.title("🌱 PREDWEEM – Predicción de Emergencia (UI Pro)")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=1.2, max_value=2.77, value=2.77, step=0.01, format="%.2f"
)
mm_window = st.sidebar.slider("Ventana media móvil (días)", 2, 14, 2, step=1)
st.sidebar.caption("Consejo: 2–4 días resalta cambios recientes; 7–14 días suaviza ruido.")

uploaded_files = st.file_uploader(
    "Sube **.xlsx** con columnas: Julian_days, TMAX, TMIN, Prec",
    type=["xlsx"], accept_multiple_files=True
)

# Cargar pesos del modelo
base = Path(".")
try:
    IW = np.load(base / "IW.npy")
    bias_IW = np.load(base / "bias_IW.npy")
    LW = np.load(base / "LW.npy")
    bias_out = np.load(base / "bias_out.npy")
except FileNotFoundError as e:
    st.error(f"Error al cargar archivos del modelo: {e}")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# Rango de visualización por defecto
fecha_inicio_def = pd.to_datetime("2025-01-15")
fecha_fin_def    = pd.to_datetime("2025-09-01")

if not uploaded_files:
    st.info("👈 Subí al menos un archivo para comenzar.")
else:
    for file in uploaded_files:
        df = pd.read_excel(file, engine="openpyxl")
        if not all(c in df.columns for c in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"{file.name} no tiene las columnas requeridas.")
            continue

        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
        pred = modelo.predict(X_real)

        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()

        # Umbrales y %
        pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / 2.77
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%) - mínimo"]      = pred["EMEAC (0-1) - mínimo"]    * 100
        pred["EMEAC (%) - máximo"]      = pred["EMEAC (0-1) - máximo"]    * 100
        pred["EMEAC (%) - ajustable"]   = pred["EMEAC (0-1) - ajustable"] * 100

        nombre = Path(file.name).stem

        # ===== Rango visible =====
        rmin = max(pred["Fecha"].min(), fecha_inicio_def)
        rmax = min(pred["Fecha"].max(), fecha_fin_def)
        c1, c2 = st.columns([1,3])
        with c1:
            st.subheader(nombre)
        with c2:
            rango = st.slider(
                "Rango a visualizar",
                min_value=pd.to_datetime(pred["Fecha"].min()).to_pydatetime(),
                max_value=pd.to_datetime(pred["Fecha"].max()).to_pydatetime(),
                value=(rmin.to_pydatetime(), rmax.to_pydatetime()),
                key=f"rango_{nombre}"
            )

        mask = (pred["Fecha"] >= rango[0]) & (pred["Fecha"] <= rango[1])
        pv = pred.loc[mask].copy()

        tabs = st.tabs(["📈 EMERREL", "📈 EMEAC (%)", "🧾 Tabla", "⬇️ Descargas"])

        # ================== EMERREL ==================
        with tabs[0]:
            pv["EMERREL_MA"] = pv["EMERREL(0-1)"].rolling(mm_window, min_periods=1).mean()

            fig1, ax1 = plt.subplots(figsize=(11.5, 4.4))
            ax1.bar(pv["Fecha"], pv["EMERREL(0-1)"], color=_colores_por_nivel(pv["Nivel_Emergencia_relativa"]))
            ax1.plot(pv["Fecha"], pv["EMERREL_MA"], linewidth=2.3)
            ax1.set_title(f"Emergencia Relativa Diaria — media móvil {mm_window}d")
            ax1.set_ylabel("EMERREL (0–1)")
            ax1.set_ylim(0, max(0.2, pv["EMERREL(0-1)"].max() * 1.1 if len(pv) else 1))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            st.pyplot(fig1, use_container_width=True)

            st.download_button("Descargar PNG (EMERREL)",
                               _fig_to_png(fig1),
                               file_name=f"{nombre}_EMERREL.png",
                               mime="image/png",
                               use_container_width=True)

        # ================== EMEAC ==================
        with tabs[1]:
            fig2, ax2 = plt.subplots(figsize=(11.5, 4.6))
            ax2.fill_between(pv["Fecha"], pv["EMEAC (%) - mínimo"], pv["EMEAC (%) - máximo"],
                             alpha=0.35, label="Banda min–max")
            ax2.plot(pv["Fecha"], pv["EMEAC (%) - ajustable"], linewidth=2.5, label="Umbral ajustable")
            ax2.plot(pv["Fecha"], pv["EMEAC (%) - mínimo"], linestyle="--", linewidth=1.2, label="Umbral mínimo")
            ax2.plot(pv["Fecha"], pv["EMEAC (%) - máximo"], linestyle="--", linewidth=1.2, label="Umbral máximo")
            for ref in [25, 50, 75, 90]:
                ax2.axhline(ref, linestyle=":", linewidth=1.0)
            ax2.set_ylabel("EMEAC (%)")
            ax2.set_ylim(0, 100)
            ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax2.legend(loc="lower right", ncols=2, framealpha=0.95)
            st.pyplot(fig2, use_container_width=True)

            cross50 = pred.loc[pred["EMEAC (%) - ajustable"] >= 50]
            if not cross50.empty:
                st.caption(f"⚑ 50% alcanzado el **{cross50.iloc[0]['Fecha'].date()}**.")

        # ================== TABLA ==================
        with tabs[2]:
            tabla = pred[["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%) - ajustable"]].copy()
            st.dataframe(
                tabla.loc[mask],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                    "Nivel_Emergencia_relativa": st.column_config.TextColumn("Nivel"),
                    "EMEAC (%) - ajustable": st.column_config.NumberColumn("EMEAC (%)", format="%.1f"),
                },
                height=370
            )

        # ================== DESCARGAS ==================
        with tabs[3]:
            csv = pred[["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]].to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv, f"{nombre}_EMEAC.csv", "text/csv", use_container_width=True)
            xlsx = _excel_export(pred[["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]], f"{nombre}_resultados.xlsx")
            st.download_button("Descargar Excel", xlsx, f"{nombre}_resultados.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
