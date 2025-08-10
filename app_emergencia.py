import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from io import BytesIO
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW; self.bias_IW = bias_IW; self.LW = LW; self.bias_out = bias_out
        self.input_min = np.array([1, 0, -7, 0]); self.input_max = np.array([300, 41, 25.5, 84])
    def tansig(self, x): return np.tanh(x)
    def normalize_input(self, X_real): return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1
    def desnormalize_output(self, y_norm, ymin=-1, ymax=1): return (y_norm - ymin) / (ymax - ymin)
    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW; a1 = self.tansig(z1); z2 = self.LW @ a1 + self.bias_out; return self.tansig(z2)
    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalize_output(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm); valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)
        def clasificar(v): return "Bajo" if v < 0.02 else ("Medio" if v <= 0.079 else "Alto")
        riesgo = np.array([clasificar(v) for v in emerrel_diff])
        return pd.DataFrame({"EMERREL(0-1)": emerrel_diff, "Nivel_Emergencia_relativa": riesgo})

st.set_page_config(page_title="PREDWEEM (UI mejorada)", layout="wide")
st.title("🌱 Predicción de Emergencia Agrícola con ANN — UI mejorada")

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input("Umbral de EMEAC para 100%", min_value=1.2, max_value=2.77, value=2.77, step=0.01, format="%.2f")

uploaded_files = st.file_uploader("Sube uno o más .xlsx con columnas: Julian_days, TMAX, TMIN, Prec", type=["xlsx"], accept_multiple_files=True)

base = Path(".")
try:
    IW = np.load(base / "IW.npy"); bias_IW = np.load(base / "bias_IW.npy")
    LW = np.load(base / "LW.npy"); bias_out = np.load(base / "bias_out.npy")
except FileNotFoundError as e:
    st.error(f"Error al cargar archivos del modelo: {e}"); st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)
def obtener_colores(niveles): return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"})
def fig_to_png_bytes(fig): buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); return buf.getvalue()

fecha_inicio_def = pd.to_datetime("2025-01-15"); fecha_fin_def = pd.to_datetime("2025-09-01")

def export_excel_formatted(df, filename):
    """Exporta a Excel. Usa xlsxwriter si está presente; si no, openpyxl sin formato."""
    buf = BytesIO()
    try:
        import xlsxwriter  # noqa: F401
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            sheet = "resultados"
            df.to_excel(writer, index=False, sheet_name=sheet)
            wb = writer.book; ws = writer.sheets[sheet]
            fmt_date = wb.add_format({"num_format": "yyyy-mm-dd"})
            fmt_pct = wb.add_format({"num_format": "0.0"})
            ws.set_column("A:A", 12, fmt_date); ws.set_column("B:B", 18); ws.set_column("C:C", 12, fmt_pct)
    except ModuleNotFoundError:
        # Fallback sin estilos (openpyxl suele estar instalado en Streamlit Cloud)
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="resultados")
    return buf.getvalue(), filename

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_excel(file, engine="openpyxl")
        if not all(col in df.columns for col in ["Julian_days", "TMAX", "TMIN", "Prec"]):
            st.warning(f"{file.name} no tiene las columnas requeridas."); continue
        X_real = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy()
        fechas = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas; pred["Julian_days"] = df["Julian_days"]; pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 2.77
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        pred["EMEAC (%) - mínimo"] = pred["EMEAC (0-1) - mínimo"] * 100
        pred["EMEAC (%) - máximo"] = pred["EMEAC (0-1) - máximo"] * 100
        pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100
        nombre = Path(file.name).stem; colores = obtener_colores(pred["Nivel_Emergencia_relativa"])
        c1, c2 = st.columns([1,3])
        with c1: st.subheader(nombre)
        with c2:
            rmin = max(pred["Fecha"].min(), fecha_inicio_def); rmax = min(pred["Fecha"].max(), fecha_fin_def)
            rango = st.slider("Rango a visualizar",
                min_value=pd.to_datetime(pred["Fecha"].min()).to_pydatetime(),
                max_value=pd.to_datetime(pred["Fecha"].max()).to_pydatetime(),
                value=(rmin.to_pydatetime(), rmax.to_pydatetime()), key=f"rango_{nombre}")
        mask = (pred["Fecha"] >= rango[0]) & (pred["Fecha"] <= rango[1]); pred_v = pred.loc[mask].copy()
        tabs = st.tabs(["📈 EMERREL", "📈 EMEAC (%)", "🧾 Tabla", "⬇️ Descargas"])
        with tabs[0]:
            pred_v["EMERREL_MA2"] = pred_v["EMERREL(0-1)"].rolling(2, min_periods=1).mean()
            fig_er, ax_er = plt.subplots(figsize=(11, 4.2))
            ax_er.bar(pred_v["Fecha"], pred_v["EMERREL(0-1)"], color=obtener_colores(pred_v["Nivel_Emergencia_relativa"]))
            ax_er.plot(pred_v["Fecha"], pred_v["EMERREL_MA2"], linewidth=2.0)
            ax_er.set_title("Emergencia Relativa Diaria (con media móvil 2d)"); ax_er.set_ylabel("EMERREL (0–1)")
            ax_er.set_ylim(0, max(0.2, pred_v["EMERREL(0-1)"].max()*1.1 if len(pred_v) else 1))
            ax_er.xaxis.set_major_locator(mdates.MonthLocator()); ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            st.pyplot(fig_er, use_container_width=True)
            st.download_button("Descargar PNG (EMERREL)", fig_to_png_bytes(fig_er),
                               file_name=f"{nombre}_EMERREL.png", mime="image/png", use_container_width=True)
        with tabs[1]:
            fig, ax = plt.subplots(figsize=(11, 4.6))
            ax.fill_between(pred_v["Fecha"], pred_v["EMEAC (%) - mínimo"], pred_v["EMEAC (%) - máximo"], alpha=0.35, label="Banda min–max")
            ax.plot(pred_v["Fecha"], pred_v["EMEAC (%) - ajustable"], linewidth=2.2, label="Umbral ajustable")
            ax.plot(pred_v["Fecha"], pred_v["EMEAC (%) - mínimo"], linestyle='--', linewidth=1.2, label="Umbral mínimo")
            ax.plot(pred_v["Fecha"], pred_v["EMEAC (%) - máximo"], linestyle='--', linewidth=1.2, label="Umbral máximo")
            for ref in [25, 50, 75, 90]: ax.axhline(ref, linestyle=':', linewidth=1.0)
            ax.set_ylabel("EMEAC (%)"); ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.legend(loc="lower right", ncols=2, framealpha=0.9); st.pyplot(fig, use_container_width=True)
        with tabs[2]:
            tabla = pred[["Fecha", "Nivel_Emergencia_relativa", "EMEAC (%) - ajustable"]].copy()
            st.dataframe(tabla.loc[mask], use_container_width=True,
                         column_config={
                             "Fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
                             "Nivel_Emergencia_relativa": st.column_config.TextColumn("Nivel"),
                             "EMEAC (%) - ajustable": st.column_config.NumberColumn("EMEAC (%)", format="%.1f")
                         }, hide_index=True, height=360)
        with tabs[3]:
            csv = pred[["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]].to_csv(index=False).encode("utf-8")
            st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv", use_container_width=True)
            excel_bytes, fname = export_excel_formatted(pred[["Fecha","Nivel_Emergencia_relativa","EMEAC (%) - ajustable"]], f"{nombre}_resultados.xlsx")
            st.download_button("Descargar Excel", excel_bytes, fname,
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
else:
    st.info("Sube al menos un archivo .xlsx para iniciar el análisis.")
