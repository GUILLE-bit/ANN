# fetch_meteobahia.py
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://meteobahia.com.ar/scripts/forecast/for-bd.xml"
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

def fetch_xml(url=URL, timeout=30, retries=3, backoff=2):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(backoff*(i+1))
    raise RuntimeError(f"Fetch failed: {last}")

def to_f(x):
    try: return float(str(x).replace(",", "."))
    except: return None

def parse_daily(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    days = root.findall(".//forecast/tabular/day")
    rows = []
    for d in days:
        fecha  = d.find("./fecha")
        tmax   = d.find("./tmax")
        tmin   = d.find("./tmin")
        precip = d.find("./precip")
        fv = fecha.get("value") if fecha is not None else None
        if not fv: 
            continue
        rows.append({
            "Fecha": pd.to_datetime(fv).normalize(),
            "TMAX": to_f(tmax.get("value") if tmax is not None else None),
            "TMIN": to_f(tmin.get("value") if tmin is not None else None),
            "Prec": to_f(precip.get("value")) if precip is not None else 0.0,
        })
    if not rows:
        raise RuntimeError("XML sin <day> válidos.")
    df = pd.DataFrame(rows).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

if __name__ == "__main__":
    xmlb = fetch_xml()
    df = parse_daily(xmlb)
    df.to_csv(OUT, index=False)
    print(f"OK -> {OUT} ({len(df)} filas)")
