# -*- coding: utf-8 -*-
"""
App: Predicción de Emergencia Agrícola (ANN) - Plotly + MA dinámica
Autor: Generado por Data Analyst (GPT)
Descripción:
- Visualización interactiva de EMERREL (0-1) con barras por nivel de riesgo y línea de media móvil (SMA/EMA).
- Cálculo de EMEAC (%) con banda mínimo-máximo y umbral ajustable por el usuario.
- Sugerencia automática de ventana de media móvil según variabilidad (Prec o EMERREL).
- Descarga de tabla de resultados.
"""
from __future__ import annotations

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict

# ---------------- Configuración de página y estilos ----------------
st.set_page_config(page_title="EMEAC/EMERREL - ANN (Plotly)", layout="wide")
px.defaults.template = "plotly_white"

# ---------------- Utilidades ----------------
NOMBRE_COL_FECHA = "Fecha"
NOMBRE_COL_EMERREL = "EMERREL(0-1)"
NOMBRE_COL_NIVEL = "Nivel_Emergencia_relativa"
NOMBRE_COL_PREC = "Prec"  # opcional

def normaliza_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura que existan y estén en el dtype esperado las columnas clave."""
    df = df.copy()
    # Fecha
    if NOMBRE_COL_FECHA in df.columns:
        df[NOMBRE_COL_FECHA] = pd.to_datetime(df[NOMBRE_COL_FECHA], errors="coerce")
    else:
        # Intento heurístico
        for candidato in ["fecha", "date", "Fecha", "FECHA"]:
            if candidato in df.columns:
                df[NOMBRE_COL_FECHA] = pd.to_datetime(df[candidato], errors="coerce")
                break
    # EMERREL
    if NOMBRE_COL_EMERREL in df.columns:
        df[NOMBRE_COL_EMERREL] = pd.to_numeric(df[NOMBRE_COL_EMERREL], errors="coerce")
    # Nivel (texto o categórico)
    if NOMBRE_COL_NIVEL in df.columns:
        df[NOMBRE_COL_NIVEL] = df[NOMBRE_COL_NIVEL].astype(str)
    # Prec opcional
    if NOMBRE_COL_PREC in df.columns:
        df[NOMBRE_COL_PREC] = pd.to_numeric(df[NOMBRE_COL_PREC], errors="coerce")

    # Orden por fecha
    if NOMBRE_COL_FECHA in df.columns:
        df = df.sort_values(NOMBRE_COL_FECHA).reset_index(drop=True)
    return df

def obtener_colores(niveles: pd.Series | None, fallback_values: pd.Series | None = None) -> list[str]:
    """
    Devuelve una lista de colores para las barras según el nivel de riesgo.
    Soporta etiquetas comunes y hace fallback a una escala por valor si no hay niveles.
    """
    mapping = {
        "bajo": "#8bc34a",
        "medio": "#ffc107",
        "alto": "#f44336",
        "muy alto": "#d32f2f",
        "critico": "#b71c1c",
        "crítico": "#b71c1c",
        "moderado": "#ff9800",
        "sin riesgo": "#9e9e9e",
    }
    colores = []
    if niveles is not None and niveles.notna().any():
        for v in niveles.fillna("").astype(str):
            key = v.strip().lower()
            color = mapping.get(key, None)
            if color is None:
                # Heurística: contiene palabras clave
                if "alto" in key and "muy" in key:
                    color = mapping["muy alto"]
                elif "alto" in key:
                    color = mapping["alto"]
                elif "med" in key:
                    color = mapping["medio"]
                elif "baj" in key:
                    color = mapping["bajo"]
                else:
                    color = "#607d8b"  # fallback
            colores.append(color)
    else:
        # Fallback: escala por valor EMERREL
        if fallback_values is None:
            fallback_values = pd.Series([0]*len(niveles) if niveles is not None else [0])
        v = pd.to_numeric(fallback_values, errors="coerce").fillna(0.0).values
        # tres tramos 0-0.33-0.66-1.0
        for x in v:
            if x < 0.33:
                colores.append("#8bc34a")
            elif x < 0.66:
                colores.append("#ffc107")
            else:
                colores.append("#f44336")
    return colores

def sugerir_ventana_ma(df: pd.DataFrame) -> Tuple[int, Dict]:
    """
    Sugiere ventana (en días) según variabilidad.
    - Si existe 'Prec': usa su desviación estándar y cuantiles para ajustar.
    - Si no: usa volatilidad de EMERREL (std de deltas).
    Devuelve (ventana_sugerida, métricas_dict).
    """
    met: Dict = {}

    if NOMBRE_COL_PREC in df.columns and df[NOMBRE_COL_PREC].notna().any():
        metrica = float(df[NOMBRE_COL_PREC].std(skipna=True))
        met["origen"] = NOMBRE_COL_PREC
        met["std"] = metrica

        vals = df[NOMBRE_COL_PREC].dropna().values
        if len(vals) >= 4:
            q_low, q_mid, q_high = np.nanpercentile(vals, [25, 50, 75])
        else:
            # Valores por defecto si hay pocos datos
            q_low, q_mid, q_high = 0.0, np.nanmean(vals) if len(vals) else 0.0, (np.nanmean(vals) if len(vals) else 1.0) * 1.5

        rango_iqr = max(q_high - q_low, 1e-9)
        # Clasificación simple
        if metrica >= rango_iqr:
            ventana = 7
        elif metrica >= (q_mid - q_low/2):
            ventana = 5
        else:
            ventana = 3
    else:
        serie = pd.to_numeric(df.get(NOMBRE_COL_EMERREL, pd.Series(dtype=float)), errors="coerce").fillna(method="ffill").fillna(0.0).values
        if len(serie) == 0:
            return 5, {"origen": "default", "std_deltas": np.nan}
        deltas = np.diff(serie, prepend=serie[0])
        vol = float(np.nanstd(deltas))
        met["origen"] = NOMBRE_COL_EMERREL
        met["std_deltas"] = vol

        if vol > 0.12:
            ventana = 7
        elif vol > 0.06:
            ventana = 5
        else:
            ventana = 3

    ventana = int(np.clip(ventana, 2, 21))
    return ventana, met

def calcular_emeac(pred: pd.DataFrame, umbral_usuario: float) -> pd.DataFrame:
    """Agrega columnas de EMEAC acumuladas y umbrales a partir de EMERREL."""
    pred = pred.copy()
    pred["EMERREL acumulado"] = pd.to_numeric(pred[NOMBRE_COL_EMERREL], errors="coerce").fillna(0.0).cumsum()

    # Umbrales (puedes ajustar las constantes si cambian los criterios)
    pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / 1.2
    pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / 2.77
    divisor = umbral_usuario if (isinstance(umbral_usuario, (int, float)) and umbral_usuario not in (0, None)) else 2.0
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / divisor

    # En porcentaje
    pred["EMEAC (%) - mínimo"]    = pred["EMEAC (0-1) - mínimo"] * 100.0
    pred["EMEAC (%) - máximo"]    = pred["EMEAC (0-1) - máximo"] * 100.0
    pred["EMEAC (%) - ajustable"] = pred["EMEAC (0-1) - ajustable"] * 100.0
    return pred

def aplicar_ma(pred: pd.DataFrame, ma_window: int, ma_tipo: str) -> pd.DataFrame:
    """Calcula columna EMERREL_MA según tipo y ventana."""
    pred = pred.copy()
    serie = pd.to_numeric(pred[NOMBRE_COL_EMERREL], errors="coerce")
    if "EMA" in ma_tipo.upper():
        alpha = 2 / (ma_window + 1)
        pred["EMERREL_MA"] = serie.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
    else:
        pred["EMERREL_MA"] = serie.rolling(window=ma_window, min_periods=1).mean()
    return pred

def fig_emerrel(pred: pd.DataFrame, nombre: str, xmin, xmax) -> go.Figure:
    colores = obtener_colores(pred.get(NOMBRE_COL_NIVEL), pred.get(NOMBRE_COL_EMERREL))
    fig = go.Figure()
    fig.add_bar(
        x=pred[NOMBRE_COL_FECHA],
        y=pred[NOMBRE_COL_EMERREL],
        marker_color=colores,
        name="EMERREL (0-1)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>EMERREL: %{y:.3f}<extra></extra>"
    )
    if "EMERREL_MA" in pred.columns:
        fig.add_scatter(
            x=pred[NOMBRE_COL_FECHA],
            y=pred["EMERREL_MA"],
            mode="lines",
            name=st.session_state.get("_ma_legend_name", "Media móvil"),
            line=dict(width=2.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>MA: %{y:.3f}<extra></extra>"
        )
    fig.update_layout(
        title=f"Emergencia Relativa Diaria - {nombre}",
        xaxis_title="Fecha",
        yaxis_title="EMERREL (0-1)",
        legend_title="Capas",
        bargap=0.15,
        hovermode="x unified"
    )
    fig.update_xaxes(range=[xmin, xmax])
    return fig

def fig_emeac(pred: pd.DataFrame, nombre: str, xmin, xmax) -> go.Figure:
    fig = go.Figure()

    # Banda min-max
    fig.add_scatter(
        x=pred[NOMBRE_COL_FECHA], y=pred["EMEAC (%) - mínimo"],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    )
    fig.add_scatter(
        x=pred[NOMBRE_COL_FECHA], y=pred["EMEAC (%) - máximo"],
        mode="lines", fill="tonexty", name="Rango entre mínimo y máximo",
        line=dict(width=0), opacity=0.35,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Rango: %{y:.1f}%<extra></extra>"
    )

    # Curvas
    fig.add_scatter(
        x=pred[NOMBRE_COL_FECHA], y=pred["EMEAC (%) - ajustable"],
        mode="lines", name="Umbral ajustable",
        line=dict(width=2.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Ajustable: %{y:.1f}%<extra></extra>"
    )
    fig.add_scatter(
        x=pred[NOMBRE_COL_FECHA], y=pred["EMEAC (%) - mínimo"],
        mode="lines", name="Umbral mínimo",
        line=dict(dash="dash", width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Mín: %{y:.1f}%<extra></extra>"
    )
    fig.add_scatter(
        x=pred[NOMBRE_COL_FECHA], y=pred["EMEAC (%) - máximo"],
        mode="lines", name="Umbral máximo",
        line=dict(dash="dash", width=1.5),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Máx: %{y:.1f}%<extra></extra>"
    )

    # Líneas horizontales
    for nivel in [25, 50, 75, 90]:
        fig.add_hline(y=nivel, line_dash="dash", line_width=1.2, opacity=0.7)

    fig.update_layout(
        title=f"Progreso EMEAC (%) - {nombre}",
        xaxis_title="Fecha",
        yaxis_title="EMEAC (%)",
        yaxis=dict(range=[0, 100]),
        legend_title="Referencias",
        hovermode="x unified"
    )
    fig.update_xaxes(range=[xmin, xmax])
    return fig

# ------------------------- App -------------------------
st.title("Predicción de Emergencia Agrícola (ANN)")

st.sidebar.header("Cargar archivos")
uploaded_files = st.sidebar.file_uploader(
    "Sube uno o más archivos (CSV o XLSX) con columnas: Fecha, EMERREL(0-1), Nivel_Emergencia_relativa, Prec (opcional).",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
umbral_usuario = st.sidebar.number_input(
    "Umbral ajustable (EMEAC)",
    min_value=0.1, max_value=10.0, value=2.0, step=0.1,
    help="Se usa en EMEAC (0-1) - ajustable = EMERREL acumulado / umbral."
)

if not uploaded_files:
    st.info("Sube al menos un archivo para comenzar.")
    st.stop()

# Procesa archivos
datasets: list[tuple[str, pd.DataFrame]] = []
for file in uploaded_files:
    nombre = os.path.splitext(os.path.basename(file.name))[0]
    try:
        if file.type.endswith("sheet") or file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            # Intenta leer CSV con auto-delimitador
            data = file.read()
            file.seek(0)
            try:
                df = pd.read_csv(io.BytesIO(data))
            except Exception:
                df = pd.read_csv(io.BytesIO(data), sep=";")
    except Exception as e:
        st.warning(f"No se pudo leer {file.name}: {e}")
        continue

    df = normaliza_columnas(df)
    if NOMBRE_COL_FECHA not in df.columns or NOMBRE_COL_EMERREL not in df.columns:
        st.warning(f"{file.name}: faltan columnas requeridas '{NOMBRE_COL_FECHA}' o '{NOMBRE_COL_EMERREL}'.")
        continue

    df["__nombre"] = nombre
    datasets.append((nombre, df))

if not datasets:
    st.error("No hay datasets válidos para mostrar.")
    st.stop()

# Selector de dataset
nombres = [n for n, _ in datasets]
idx = st.sidebar.selectbox("Selecciona dataset", options=list(range(len(nombres))), format_func=lambda i: nombres[i], index=0)
nombre, pred = datasets[idx]
pred = pred.copy()

# Rango de fechas
fecha_min = pd.to_datetime(pred[NOMBRE_COL_FECHA].min()).date()
fecha_max = pd.to_datetime(pred[NOMBRE_COL_FECHA].max()).date()
st.sidebar.subheader("Rango de fechas")
fecha_inicio, fecha_fin = st.sidebar.date_input(
    "Selecciona intervalo",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)
# Asegura tipos datetime
fecha_inicio = pd.to_datetime(fecha_inicio)
fecha_fin = pd.to_datetime(fecha_fin)

# Calcula sugerencia de ventana según rango actual
pred_filtrado_sug = pred[(pred[NOMBRE_COL_FECHA] >= fecha_inicio) & (pred[NOMBRE_COL_FECHA] <= fecha_fin)].copy()
if pred_filtrado_sug.empty:
    pred_filtrado_sug = pred.copy()
sugerida, met = sugerir_ventana_ma(pred_filtrado_sug)

# Sidebar: controles de MA
st.sidebar.markdown("---")
st.sidebar.subheader("Suavizado (Media móvil)")

auto_ma = st.sidebar.toggle(
    "Usar sugerencia automática",
    value=True,
    help="Si está activo, la ventana se fija según la variabilidad reciente (Prec o EMERREL)."
)

ma_tipo = st.sidebar.selectbox(
    "Tipo de media",
    options=["SMA (simple)", "EMA (exponencial)"],
    index=0
)

if auto_ma:
    ma_window = sugerida
    metr_val = met.get('std', met.get('std_deltas', np.nan))
    info = f"Ventana sugerida: **{ma_window} días** (basada en **{met.get('origen', 'EMERREL')}**"
    if np.isfinite(metr_val):
        info += f"; STD: {metr_val:.3f})"
    else:
        info += ")"
    st.sidebar.info(info)
else:
    ma_window = st.sidebar.slider(
        "Ventana media móvil (días)",
        min_value=2, max_value=21, value=sugerida, step=1,
        help="Puedes partir de la sugerencia y ajustarla manualmente."
    )

# Prepara dataframe principal: EMEAC + MA
pred = calcular_emeac(pred, umbral_usuario)
pred = aplicar_ma(pred, int(ma_window), ma_tipo)

# Límite de fechas seguros
xmin = max(fecha_inicio, pred[NOMBRE_COL_FECHA].min())
xmax = min(fecha_fin,     pred[NOMBRE_COL_FECHA].max())

# Guarda nombre de leyenda MA en session_state (para que figure lo muestre correctamente)
st.session_state["_ma_legend_name"] = f"Media móvil {('EMA' if 'EMA' in ma_tipo.upper() else 'SMA')} ({ma_window}d)"

# Tabs
tab_er, tab_emeac, tab_tabla = st.tabs(["EMERREL (0-1)", "EMEAC (%)", "Datos"])

with tab_er:
    st.subheader(f"EMERREL (0-1) - {nombre}")
    # Filtra para rango de visualización
    mask = (pred[NOMBRE_COL_FECHA] >= xmin) & (pred[NOMBRE_COL_FECHA] <= xmax)
    pred_viz = pred.loc[mask].copy()
    if pred_viz.empty:
        st.warning("No hay datos en el rango seleccionado.")
    else:
        fig_er = fig_emerrel(pred_viz, nombre, xmin, xmax)
        st.plotly_chart(fig_er, use_container_width=True)

with tab_emeac:
    st.subheader(f"EMEAC (%) - {nombre}")
    mask = (pred[NOMBRE_COL_FECHA] >= xmin) & (pred[NOMBRE_COL_FECHA] <= xmax)
    pred_viz = pred.loc[mask].copy()
    if pred_viz.empty:
        st.warning("No hay datos en el rango seleccionado.")
    else:
        fig_ac = fig_emeac(pred_viz, nombre, xmin, xmax)
        st.plotly_chart(fig_ac, use_container_width=True)

with tab_tabla:
    st.subheader(f"Datos calculados - {nombre}")
    columnas = [NOMBRE_COL_FECHA, NOMBRE_COL_NIVEL, NOMBRE_COL_EMERREL, "EMERREL_MA", "EMEAC (%) - ajustable"]
    columnas = [c for c in columnas if c in pred.columns]
    st.dataframe(pred[columnas], use_container_width=True)

    # Exportar con nombre de MA explícito
    pred_export = pred[columnas].rename(columns={"EMERREL_MA": f"EMERREL_MA_{('EMA' if 'EMA' in ma_tipo.upper() else 'SMA')}_{int(ma_window)}d"})
    csv = pred_export.to_csv(index=False).encode("utf-8")
    st.download_button(f"Descargar CSV - {nombre}", csv, f"{nombre}_EMEAC.csv", "text/csv")
