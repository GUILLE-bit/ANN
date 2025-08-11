name: fetch-meteo
on:
  schedule:
    - cron: "0 */6 * * *"   # cada 6 horas (UTC)
  workflow_dispatch:
permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }

      - name: Safe git dir (evita 'dubious ownership')
        run: git config --global --add safe.directory "$GITHUB_WORKSPACE"

      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }

      - run: pip install pandas requests

      # --- Diagnóstico rápido del endpoint (cabeceras + HTTP code) ---
      - name: Probe endpoint (HEAD)
        run: |
          set -e
          curl -sS -I "https://meteobahia.com.ar/scripts/forecast/for-bd.xml" || true

      # --- Ejecuta el fetch: genera data/meteo_daily.csv ---
      - name: Run fetch
        run: python fetch_meteobahia.py

      # Guarda como artefacto (útil para inspeccionar si algo falla)
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: meteo_csv
          path: data/meteo_daily.csv
          if-no-files-found: error

      # --- Publica a gh-pages con una acción estable ---
      - name: Publish to gh-pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_branch: gh-pages
          publish_dir: data
          # Dejará disponible: https://GUILLE-bit.github.io/ANN/meteo_daily.csv
