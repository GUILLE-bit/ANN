import time, requests, pandas as pd, numpy as np, xml.etree.ElementTree as ET
from pathlib import Path

URL_FCST = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
OUT = Path("data/meteo_daily.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0.0.0 Safari/537.36"),
    "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://meteobahia.com.ar/",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.8",
}

def fetch_fcst_xml(url=URL_FCST, timeout=30, retries=3, backoff=2):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(backoff*(i+1))
    raise RuntimeError(f"Fetch forecast failed: {last}")

def parse_fcst(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    days = root.findall(".//forecast/tabular/day")
    rows = []
    def to_f(x):
        try: return float(str(x).replace(",", "."))
        except: return None
    for d in days:
        fecha  = d.find("./fecha")
        tmax   = d.find("./tmax")
        tmin   = d.find("./tmin")
        precip = d.find("./precip")
        fval = fecha.get("value") if fecha is not None else None
        if not fval: 
            continue
        rows.append({
            "Fecha": pd.to_datetime(fval).normalize(),
            "TMAX": to_f(tmax.get("value") if tmax is not None else None),
            "TMIN": to_f(tmin.get("value") if tmin is not None else None),
            "Prec": to_f(precip.get("value")) if precip is not None else 0.0,
        })
    if not rows:
        raise RuntimeError("Forecast XML sin <day> válidos.")
    df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

# ----------------- HISTÓRICO tolerante (xlsx/xlsx.xlsx/csv) -----------------
CANDIDATOS = [
    Path("data/historico.xlsx"),
    Path("data/historico.xlsx.xlsx"),
    Path("data/historico.csv"),
]

def _to_float_series(s):
    if s is None: return None
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def load_hist_any(target_year: int, first_fcst_day: pd.Timestamp) -> pd.DataFrame:
    ruta = next((p for p in CANDIDATOS if p.exists()), None)
    if not ruta:
        print("[hist] ⚠️ No hay histórico en data/. Sigo solo con pronóstico.")
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    # Leer
    if ruta.suffix.lower() == ".csv":
        df = pd.read_csv(ruta)
    else:
        df = pd.read_excel(ruta)  # requiere openpyxl
    print(f"[hist] Usando {ruta} | filas={len(df)} | columnas={list(df.columns)}")

    # Normalizar nombres
    cols_map = {c.lower().strip(): c for c in df.columns}
    fecha_key = next((k for k in cols_map if k in ("fecha","date")), None)
    doy_key   = next((k for k in cols_map if k in ("julian_days","julian day","dia_juliano","dia juliano","doy")), None)
    # T variables
    def find_name(cands): return next((cols_map[k] for k in cols_map if k in cands), None)
    tmax_col = find_name({"tmax","t max","temp_max","temp max","t.max"})
    tmin_col = find_name({"tmin","t min","temp_min","temp min","t.min"})
    prec_col = find_name({"prec","precip","precipitacion","precipitación","lluvia","pp","pr"})

    if fecha_key:
        df = df.rename(columns={cols_map[fecha_key]: "Fecha"})
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce").dt.normalize()
        df["Julian_days"] = df["Fecha"].dt.dayofyear
    elif doy_key:
        df = df.rename(columns={cols_map[doy_key]: "Julian_days"})
        base = pd.Timestamp(f"{target_year}-01-01")
        df["Fecha"] = base + pd.to_timedelta(df["Julian_days"].astype(int) - 1, unit="D")
    else:
        raise ValueError("El histórico debe tener 'Fecha' o 'Julian_days'.")

    if tmax_col: df = df.rename(columns={tmax_col: "TMAX"})
    if tmin_col: df = df.rename(columns={tmin_col: "TMIN"})
    if prec_col: df = df.rename(columns={prec_col: "Prec"})

    for c in ("TMAX","TMIN","Prec"):
        if c in df.columns:
            df[c] = _to_float_series(df[c])

    req = {"Fecha","Julian_days","TMAX","TMIN","Prec"}
    faltan = req - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en histórico: {', '.join(sorted(faltan))}")

    df = df.dropna(subset=["Fecha"]).sort_values("Fecha")
    # --- Filtro por año del pronóstico ---
    df_year = df[df["Fecha"].dt.year == target_year].copy()

    # Fallback: si no hay filas del año target pero sí hay fechas de otro año,
    # recalculamos Fecha a partir del DOY para el año del pronóstico.
    if df_year.empty and len(df):
        print("[hist] ⚠️ Histórico sin filas para el año target. Reasigno año por DOY.")
        if "Julian_days" not in df.columns:
            df["Julian_days"] = df["Fecha"].dt.dayofyear
        base = pd.Timestamp(f"{target_year}-01-01")
        df["Fecha"] = base + pd.to_timedelta(df["Julian_days"].astype(int) - 1, unit="D")
        df_year = df.copy()

    # Hasta el día previo al primer pronóstico
    df_year = df_year[df_year["Fecha"] < first_fcst_day]
    df_year = df_year.drop_duplicates(subset=["Fecha"], keep="last")
    print(f"[hist] Rango final: {df_year['Fecha'].min()} → {df_year['Fecha'].max()} | filas={len(df_year)}")
    return df_year[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def main():
    # Pronóstico
    xmlb = fetch_fcst_xml()
    df_fcst = parse_fcst(xmlb)
    first_fcst_day = df_fcst["Fecha"].min()
    year = int(first_fcst_day.year)
    print(f"[fcst] Rango: {df_fcst['Fecha'].min().date()} → {df_fcst['Fecha'].max().date()} | filas={len(df_fcst)}")

    # Histórico
    df_hist = load_hist_any(year, first_fcst_day)

    # Unir
    df_all = pd.concat([df_hist, df_fcst], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Fecha"], keep="first").sort_values("Fecha").reset_index(drop=True)
    df_all["Julian_days"] = df_all["Fecha"].dt.dayofyear

    # Guardar
    df_all.to_csv(OUT, index=False)
    print(f"[OK] Guardado {OUT} con {len(df_all)} filas | {df_all['Fecha'].min().date()} → {df_all['Fecha'].max().date()}")

if __name__ == "__main__":
    main()
