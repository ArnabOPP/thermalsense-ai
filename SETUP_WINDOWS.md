# Person A — Complete Windows Setup Guide
## From zero to running the pipeline

---

## Step 0 — Do these RIGHT NOW (before anything else)

### 0.1 Register for Google Earth Engine
1. Go to https://earthengine.google.com/
2. Click **Sign Up** → use your Google account
3. Select **Non-commercial / Academic**
4. Wait for approval email (usually 1–2 days, sometimes minutes)
5. Once approved, note your **GCP Project ID** — you'll need it in Step 4

### 0.2 Register for MOSDAC (ISRO's data portal)
1. Go to https://mosdac.gov.in/
2. Click **Register** in the top right
3. Fill in the form — use your institutional email
4. Activation is instant
5. Note your username and password

### 0.3 Clone the repo
```
git clone https://github.com/your-team/thermalsense.git
cd thermalsense/data-pipeline
```
If you don't have git: https://git-scm.com/download/win

---

## Step 1 — Install Miniconda (Python package manager)

Miniconda is lighter than Anaconda and handles geospatial libraries much better on Windows.

1. Download: https://docs.conda.io/en/latest/miniconda.html
   → Choose **Miniconda3 Windows 64-bit**

2. Run the installer. When asked:
   - ✅ "Add Miniconda3 to my PATH" — check this box
   - ✅ "Register Miniconda3 as my default Python"

3. Open **Anaconda Prompt** (search in Start menu) — use this for everything below, NOT regular CMD

4. Verify:
   ```
   conda --version
   python --version
   ```
   You should see conda 23.x and Python 3.x

---

## Step 2 — Create the thermalsense environment

This installs all geospatial libraries in an isolated environment.
Run in Anaconda Prompt from the `data-pipeline` folder:

```bash
conda env create -f environment.yml
```

This will take **10–20 minutes** — it's downloading ~2GB of packages. Don't close the window.

Activate the environment (do this every time you open a new terminal):
```bash
conda activate thermalsense
```

Verify everything installed:
```bash
python -c "import ee, rasterio, geopandas, osmnx, torch; print('All imports OK')"
```

---

## Step 3 — Install GDAL command-line tools

GDAL provides `gdalwarp` which scripts 01–05 use for reprojection.
With conda it's already installed. Verify:

```bash
gdalwarp --version
```

You should see: `GDAL 3.8.x, released 2023/...`

If not found:
```bash
conda install -c conda-forge gdal
```

---

## Step 4 — Configure your credentials

Copy the example env file:
```bash
# In Anaconda Prompt, inside data-pipeline/
copy .env.example .env
```

Open `.env` in Notepad and fill in:
```
EE_PROJECT_ID=your-gcp-project-id
MOSDAC_USERNAME=your_mosdac_username
MOSDAC_PASSWORD=your_mosdac_password
```

Finding your GCP Project ID:
1. Go to https://console.cloud.google.com/
2. Top bar shows your project name → click it
3. Copy the **Project ID** (not the name — they're different)

---

## Step 5 — Authenticate Google Earth Engine

Run this once. It opens a browser window:

```bash
conda activate thermalsense
python -c "import ee; ee.Authenticate()"
```

1. A browser opens → sign in with your Google account
2. Click **Allow**
3. Copy the authorization code shown
4. Paste it back in the terminal
5. Done — credentials are cached. You won't need to do this again.

Test authentication:
```bash
python -c "
import ee, os
from dotenv import load_dotenv
load_dotenv('.env')
ee.Initialize(project=os.environ['EE_PROJECT_ID'])
info = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_138044_20230315').getInfo()
print('GEE working! Image date:', info['properties']['DATE_ACQUIRED'])
"
```

---

## Step 6 — Verify the full setup

```bash
conda activate thermalsense
python scripts/utils.py
```

Expected output: no errors (the utils module just imports and exits cleanly).

---

## Step 7 — Run the pipeline for Kolkata

### Quick test (1 year, 1 season — takes ~15 min):
```bash
python run_pipeline.py --city kolkata --year 2024 --season pre_monsoon
```

### Full run (all years — takes ~90 min):
```bash
python run_pipeline.py --city kolkata
```

### If MOSDAC isn't working yet, use ERA5 fallback:
```bash
python run_pipeline.py --city kolkata --use-era5
```

### Run individual scripts:
```bash
python scripts/01_pull_landsat_lst.py --city kolkata --year 2024
python scripts/02_pull_sentinel2.py --city kolkata --year 2024
python scripts/03_osm_morphology.py --city kolkata
python scripts/04_mosdac_insat3d.py --city kolkata --use-era5
python scripts/05_feature_engineering.py --city kolkata --year 2024
python scripts/06_quality_check.py --city kolkata
```

---

## Step 8 — Add a new city

Adding any Indian city is one step:

1. Open `config/config.yaml`
2. Add an entry under `cities:` with the city's bounding box:

```yaml
cities:
  chennai:
    display_name: "Chennai"
    bbox: [80.19, 12.90, 80.34, 13.18]
    utm_epsg: 32644
    landsat_path: 142
    landsat_row: 52
    admin_level: 9
    cpcb_stations: []
```

3. Run:
```bash
python run_pipeline.py --city chennai
```

That's it. The pipeline auto-configures everything else.

Finding bbox values:
- Go to https://bboxfinder.com/
- Draw a rectangle around your city
- Copy the [West, South, East, North] values

Finding Landsat path/row:
- Go to https://landsat.usgs.gov/pathrow-shapefiles
- Or use: https://www.usgs.gov/landsat-missions/wrs-2-pathrow-files

Finding UTM EPSG:
- Go to https://epsg.io/
- Search your city → check the UTM zone for that longitude

---

## Common errors and fixes

### `conda: command not found`
→ Use Anaconda Prompt (from Start menu), not regular Command Prompt

### `EEException: Earth Engine not initialized`
→ Run: `python -c "import ee; ee.Authenticate()"`

### `GDAL not found` or `gdalwarp not found`
→ Run: `conda install -c conda-forge gdal`
→ Then restart Anaconda Prompt

### `ModuleNotFoundError: No module named 'ee'`
→ You forgot to activate: `conda activate thermalsense`

### GEE download timeout / connection error
→ The script retries 3 times automatically
→ If still failing, check your internet connection
→ Try again later (GEE has rate limits)

### `OSMnx fetch failed`
→ Overpass API (OSM's query server) has occasional downtime
→ Wait 15 minutes and retry
→ The script will use cached OSM data on retry

### MOSDAC login fails
→ Check credentials in .env
→ MOSDAC may be undergoing maintenance — use `--use-era5` flag

---

## Output files (hand these to Person B)

After a full run, these files will exist:

```
outputs/
├── processed/kolkata/
│   ├── landsat/     landsat_lst_{year}_{season}_utm32645.tif  ← LST maps
│   ├── sentinel2/   s2_indices_{year}_{season}_utm32645.tif   ← NDVI etc.
│   ├── morphology/  morphology_kolkata_utm32645.tif            ← Urban form
│   └── insat3d/     insat3d_tatm_{year}_{season}_utm32645.tif ← Atmo. temp
└── exports/kolkata/
    ├── feature_matrix_kolkata_2024_pre_monsoon.parquet   ← Per season
    └── feature_matrix_kolkata_ALL.parquet                ← ALL YEARS — give this to Person B
```

**Tell Person B:**
> "Feature matrix is ready at outputs/exports/kolkata/feature_matrix_kolkata_ALL.parquet
> It has N rows × 22 columns. LST target is 'lst_celsius'. Run: python model/xgb_baseline.py"

---

## Questions / stuck?

1. Check `logs/` — every script writes a detailed log file
2. Run quality check: `python scripts/06_quality_check.py --city kolkata`
3. Check the metadata sidecar files: each .tif has a .meta.json next to it
