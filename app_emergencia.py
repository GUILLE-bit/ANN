# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="PREDICCION EMERGENCIA AGRICOLA LOLIUM SP", layout="wide")

# =================== Modelo ANN ===================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out, low=0.02, medium=0.079):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = bias_out
        # Orden esperado: [Julian_days, TMAX, TMIN, Prec]
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
        self.low_thr = low
        self.med_thr = medium

    def tansig(self, x):
        return np.tanh(x)

    def normalize_input(self, X_real):
        return 2 * (X_real - self.input_min) / (self.input_max - self.input_min) - 1

    def desnormalizar_salida(self, y_norm, ymin=-1, ymax=1):
        # Mapea de [-1, 1] a [0, 1]
        return (y_norm - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2)

    def _clasificar(self, valor):
        if valor < self.low_thr:
            return "Bajo"
        elif valor <= self.med_thr:
            return "Medio"
        else:
            return "Alto"

    def predict(self, X_real):
        X_norm = self.normalize_input(X_real)
        emerrel_pred = np.array([self._predict_single(x) for x in X_norm])
        emerrel_desnorm = self.desnormalizar_salida(emerrel_pred)
        emerrel_cumsum = np.cumsum(emerrel_desnorm)
        valor_max_emeac = 8.05
        emer_ac = emerrel_cumsum / valor_max_emeac
        emerrel_diff = np.diff(emer_ac, prepend=0)

        riesgo = np.array([self._clasificar(v) for v in emerrel_diff])

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

@st.cache_data(show_spinner=False)
def load_weights(base_dir: Path):
    IW = np.load(base_dir / "IW.npy")
    bias_IW = np.load(base_dir / "bias_IW.npy")
    LW = np.load(base_dir / "LW.npy")
    bias_out = np.load(base_dir / "bias_out.npy")
    return IW, bias_IW, LW, bias_out

# =================== UI ===================
st.title("PREDICCION EMERGENCIA AGRICOLA LOLIUM SP")

st.sidebar.header("Fuente de datos")
fuente = st.sidebar.radio("Elegí la fuente", ["Automático (CSV público)", "Subir Excel"])

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input(
    "Umbral de EMEAC para 100%",
    min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f"
)

st.sidebar.header("Validaciones")
mostrar_fuera_rango = st.sidebar.checkbox("Avisar datos fuera de rango de entrenamiento", value=False)

# Botón para forzar recarga de datos cacheados
if st.sidebar.button("Forzar recarga de datos"):
    st.cache_data.clear()

# Cargar pesos del modelo
try:
    base = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    IW, bias_IW, LW, bias_out = load_weights(base)
except FileNotFoundError as e:
    st.error(
        "Error al cargar archivos del modelo (IW.npy, bias_IW.npy, LW.npy, bias_out.npy). "
        f"Ruta buscada: {base}. Detalle: {e}"
    )
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
            # Coerción a numérico por robustez
            cols = ["Julian_days", "TMAX", "TMIN", "Prec"]
            df_up[cols] = df_up[cols].apply(pd.to_numeric, errors="coerce")
            bad = df_up[cols].isna().any(axis=1).sum()
            if bad:
                st.warning(f"{file.name}: {bad} filas con valores inválidos fueron excluidas.")
                df_up = df_up.dropna(subset=cols)

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

        if mostrar_fuera_rango and detectar_fuera_rango(X_real, modelo.input_min, modelo.input_max):
            st.info(f"⚠️ {nombre}: hay valores fuera del rango de entrenamiento ({modelo.input_min} a {modelo.input_max}).")

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
            yr = int(st.sidebar.selectbox(
                "Año a mostrar (reinicio 1/feb → 1/sep)",
                sorted(years),
                key=f"year_select_{nombre}"  # clave única por dataset
            ))

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

        # --------- Gráfico 1: EMERGENCIA RELATIVA DIARIA (Plotly interactivo) ---------
        st.subheader("EMERGENCIA RELATIVA DIARIA")

        colores_list = colores_vis.tolist()
        fig_er = go.Figure()

        # Barras EMERREL(0-1)
        fig_er.add_bar(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL(0-1)"],
            marker=dict(color=colores_list),
            hovertemplate=(
                "Fecha: %{x|%d-%b-%Y}<br>"
                "EMERREL: %{y:.3f}<br>"
                "Nivel: %{customdata}"
            ),
            customdata=pred_vis["Nivel_Emergencia_relativa"],
            name="EMERREL (0-1)",
        )

        # Línea media móvil 5 días
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5_rango"],
            mode="lines",
            name="Media móvil 5 días (rango)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
        ))

        # Área bajo la media móvil (fill to zero)
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5_rango"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            name="Área MA5",
            hoverinfo="skip",
            showlegend=False
        ))

        # ====== Líneas horizontales de referencia de niveles (con leyenda) ======
        low_thr = float(modelo.low_thr)
        med_thr = float(modelo.med_thr)

        # Línea de referencia 'Bajo'
        fig_er.add_trace(go.Scatter(
            x=[fecha_inicio_rango, fecha_fin_rango],
            y=[low_thr, low_thr],
            mode="lines",
            line=dict(color="green", dash="dot"),
            name=f"Bajo (≤ {low_thr:.3f})",
            hoverinfo="skip"
        ))
        # Línea de referencia 'Medio'
        fig_er.add_trace(go.Scatter(
            x=[fecha_inicio_rango, fecha_fin_rango],
            y=[med_thr, med_thr],
            mode="lines",
            line=dict(color="orange", dash="dot"),
            name=f"Medio (≤ {med_thr:.3f})",
            hoverinfo="skip"
        ))
        # Entrada de leyenda para 'Alto' (sin línea fija)
        fig_er.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="red", dash="dot"),
            name=f"Alto (> {med_thr:.3f})",
            hoverinfo="skip",
            showlegend=True
        ))

        fig_er.update_layout(
            title="EMERGENCIA RELATIVA DIARIA",
            xaxis_title="Fecha",
            yaxis_title="EMERREL (0-1)",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend_title="Referencias"
        )
        fig_er.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")
        fig_er.update_yaxes(rangemode="tozero")

        st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

        # --------- Gráfico 2: EMEAC (rango) - Plotly interactivo ---------
        st.subheader("EMERGENCIA ACUMULADA DIARIA")

        fig = go.Figure()

        # Banda entre máximo y mínimo (usar fill entre trazos)
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - máximo (rango)"],
            mode="lines",
            line=dict(width=0),
            name="Rango entre mínimo y máximo (reiniciado)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - mínimo (rango)"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>",
            showlegend=False
        ))

        # Líneas de umbrales
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable (rango)"],
            mode="lines", name="Umbral ajustable (reiniciado)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
            mode="lines", name="Umbral mínimo (reiniciado)", line=dict(dash="dash"),
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
            mode="lines", name="Umbral máximo (reiniciado)", line=dict(dash="dash"),
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))

        # Líneas horizontales 25, 50, 75, 90 %
        for nivel in [25, 50, 75, 90]:
            fig.add_hline(y=nivel, line_dash="dash", opacity=0.6)

        fig.update_layout(
            title="EMERGENCIA ACUMULADA DIARIA",
            xaxis_title="Fecha",
            yaxis_title="EMEAC (%)",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
            legend_title="Referencias",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig.update_xaxes(range=[fecha_inicio_rango, fecha_fin_rango], dtick="M1", tickformat="%b")

        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

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

