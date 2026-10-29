# Data Pipeline

## Sources

| Dataset | Purpose | Native sampling | Acquisition |
|---|---|---|---|
| ERA5 | Surface wind at 10 m | 0.25°, hourly | Copernicus CDS |
| HYCOM | Ocean current `u/v` | ~0.04° × 0.08°, 3-hourly | HYCOM THREDDS/NCSS |
| IBTrACS | TC center, time, wind, pressure | mostly 3/6-hourly | NOAA NCEI |

The code does not redistribute these products. Users are responsible for
following the providers' terms and citation requirements.

## CDS Credentials

Create `~/.cdsapirc` in the user's home directory:

```text
url: https://cds.climate.copernicus.eu/api
key: <YOUR_KEY>
```

Never place credentials in this repository. The former
`cds_api_keys.json` file is removed from the current tree and ignored.

## Required Archive

```text
data/
├── union_region/<TC_TAG>/hycom_uv_<depth>m.nc
├── era5_wind/<TC_TAG>/era5_10m_wind_hourly.nc
├── typhoon_split_500.json
└── segments_500.csv
```

The final underwater depths are:

```text
0, 10, 20, 30, 40, 50, 60 m
```

## Event Split and Segmentation

- 500 western North Pacific TCs.
- 450 training TCs and 50 held-out test TCs, separated at event level.
- Tracks interpolated to 10 min.
- Target segment length: 300–500 km.
- Segment stride: 250 km.
- Results: 7,248 segments from 495 usable TCs; 768 held-out test segments.

Generate or validate the split and segments:

```bash
python slice_segments_500.py
python audit_hycom_currents.py \
  --local-root data/union_region \
  --nas-root /path/to/nas/union_current_depths \
  --print-missing-depths
```

## NAS-Backed Data

Large NetCDF files may remain on a NAS. Expose them under the expected paths:

```bash
ln -s /nas/archive/STORM/hycom_uv_30m.nc \
  data/union_region/STORM/hycom_uv_30m.nc
```

For repeated multi-process training, build an mmap cache on local storage:

```bash
export UWFLOW_CACHE_DIR="$HOME/.cache/auv_tc/uwflow"
python build_uwflow_cache.py 30 0.25 3
```

Training reads the cache instead of repeatedly traversing the NAS.
