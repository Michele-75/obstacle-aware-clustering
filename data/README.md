# Data Directory

## Structure

- **`raw/`** — Original, unmodified data files (hotel listings, environmental monitoring data)
- **`processed/`** — Cleaned, normalized datasets ready for clustering
- **`boundaries/`** — Obstacle boundary coordinates (ellipse outline, lake shoreline)

## Data Sources

| File | Source | Description |
|------|--------|-------------|
| `raw/hotels_formatted.txt` | Google Maps Places API | Hotels around Lake Tahoe with lat/lon, rating, price |
| `boundaries/ellipse_scaled_xy.txt` | Generated | Scaled ellipse boundary for the toy problem |
| `boundaries/snapped_path_lake_tahoe.txt` | Google Maps Roads API | Snapped boundary points for Lake Tahoe |

## Reproducibility

Raw data files are included in this repository for reproducibility. If you need to re-collect data, see the collection notebooks and scripts in `notebooks/` and `src/`.
