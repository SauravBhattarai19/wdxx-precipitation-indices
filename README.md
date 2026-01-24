# Global Extreme Precipitation Analysis: WDXX and WDXXR Indices

This repository contains all computational codes for the analysis of precipitation concentration indices, including the novel WDXXR (chronological clustering) indices and multi-threshold concentration analysis.

## Repository Structure

```
Global_Extreme_Analysis/
├── data_processing/              # Data processing scripts
│   ├── 00_calculate_precipitation_percentiles.py
│   ├── 08_ETCCDI_Precipitation_Indices.py
│   ├── 09_enhanced_precipitation_concentration_indices.py
│   └── 10_WD50R_Chronological_Precipitation_Index.py
├── scripts/                      # Utility scripts
│   ├── combine_wdxx_indices.py   # Combine WDXX files (WD25, WD50, WD75)
│   └── combine_wdxxr_indices.py  # Combine WDXXR files (WD25R, WD50R, WD75R)
├── visualization/                # Visualization scripts
│   ├── baseline_precipitation.py
│   ├── etccdi_trends.py
│   ├── figure3.py
│   ├── median_change.py
│   ├── prcptotwdxxr.py
│   ├── ratio_difference.py
│   ├── trends.py
│   ├── wdxx_trends.py
│   └── wdxxr_trends.py
├── README.md                      # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

## Overview

This repository implements:

1. **Precipitation Percentiles Calculation** (`00_calculate_precipitation_percentiles.py`)
   - Calculates precipitation percentiles from ERA5 data
   - Required for subsequent analysis

2. **ETCCDI Precipitation Indices** (`08_ETCCDI_Precipitation_Indices.py`)
   - Standard ETCCDI indices including WD50
   - Multi-threshold analysis (WD25, WD50, WD75)

3. **Enhanced Precipitation Concentration Indices** (`09_enhanced_precipitation_concentration_indices.py`)
   - Advanced concentration metrics beyond basic WD50
   - Includes PCF, EPD, PGC, PE, and regime classification
   - Multi-threshold analysis (WD10, WD25, WD50, WD75, WD90)

4. **WD50R Chronological Precipitation Index** (`10_WD50R_Chronological_Precipitation_Index.py`)
   - Novel chronological clustering indices (WD25R, WD50R, WD75R)
   - Considers temporal sequence of precipitation events
   - Regime-specific analysis

5. **Combination Scripts**
   - `combine_wdxx_indices.py`: Combines annual WDXX files into time-series
   - `combine_wdxxr_indices.py`: Combines annual WDXXR files into time-series

6. **Visualization Scripts**
   - Publication-quality figures for manuscript
   - Trend analysis and comparison plots
   - Spatial and temporal visualizations

## Data Requirements

### Input Data
- **ERA5 daily precipitation data** (NetCDF format) from the **Copernicus Climate Change Service (C3S) Climate Data Store**
  - Data source: https://cds.climate.copernicus.eu/
  - Files should be named: `era5_daily_{year}_{month:02d}.nc`
  - Expected location: `/data/climate/disk1/datasets/era5/` (configurable via command-line arguments)
  - All scripts in `data_processing/` require ERA5 data from C3S as input

### Output Data
- Processed indices saved to `data/processed/` directory
- Organized by index type (etccdi_indices, enhanced_concentration_indices, wd50r_indices)

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.7+
- xarray
- numpy
- pandas
- scipy
- matplotlib
- cartopy
- netCDF4
- regionmask (optional, for land/ocean masking)

## Usage

### Step 1: Calculate Precipitation Percentiles

```bash
python data_processing/00_calculate_precipitation_percentiles.py \
    --start-year 1980 --end-year 2024 \
    --precip-dir /path/to/era5/data \
    --output-dir data/processed/percentiles
```

### Step 2: Calculate ETCCDI Indices

```bash
python data_processing/08_ETCCDI_Precipitation_Indices.py \
    --start-year 1980 --end-year 2024 \
    --precip-dir /path/to/era5/data \
    --output-dir data/processed/etccdi_indices
```

### Step 3: Calculate Enhanced Concentration Indices

```bash
python data_processing/09_enhanced_precipitation_concentration_indices.py \
    --start-year 1980 --end-year 2024 \
    --method all \
    --precip-dir /path/to/era5/data \
    --output-dir data/processed/enhanced_concentration_indices
```

### Step 4: Calculate WD50R Chronological Indices

```bash
python data_processing/10_WD50R_Chronological_Precipitation_Index.py \
    --start-year 1980 --end-year 2024 \
    --percentiles 25 50 75 \
    --precip-dir /path/to/era5/data \
    --output-dir data/processed/wd50r_indices
```

### Step 5: Combine Annual Files into Time-Series

```bash
# Combine WDXX indices (WD25, WD50, WD75)
python scripts/combine_wdxx_indices.py \
    --input-dir data/processed/enhanced_concentration_indices \
    --output-file data/processed/wdxx_1980_2024.nc \
    --start-year 1980 --end-year 2024 \
    --method all

# Combine WDXXR indices (WD25R, WD50R, WD75R)
python scripts/combine_wdxxr_indices.py \
    --input-dir data/processed/wd50r_indices \
    --output-file data/processed/wdxxr_1980_2024.nc \
    --start-year 1980 --end-year 2024
```

### Step 6: Generate Visualizations

```bash
# Generate publication figures
python visualization/figure3.py \
    --etccdi-dir data/processed/etccdi_indices \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --mask-type land

python visualization/trends.py \
    --etccdi-dir data/processed/etccdi_indices \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --mask-type land
```

## Manuscript Figures

This section documents how each figure in the manuscript was generated, including the exact command-line codes used.

### Main Figures

#### Figure 2: Multi-Threshold Period Comparison

**Filename:** `figures/multi_threshold_period_comparison.png`  
**Script:** `visualization/median_change.py`  
**Description:** Compares median values of WD25, WD50, and WD75 between two periods (1980-1999 vs 2000-2024) with relative change percentages.

**Command:**
```bash
python visualization/median_change.py \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --output-dir figures/ \
    --include-multi WD25 WD50 WD75 \
    --skip-etccdi \
    --skip-chrono \
    --mask-type land \
    --period1-start 1980 \
    --period1-end 1999 \
    --period2-start 2000 \
    --period2-end 2024
```

#### Figure 3: Chronological Period Comparison

**Filename:** `figures/chronological_period_comparison.png`  
**Script:** `visualization/median_change.py`  
**Description:** Compares median values of WD25R, WD50R, and WD75R between two periods (1980-1999 vs 2000-2024) with relative change percentages.

**Command:**
```bash
python visualization/median_change.py \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --include-chrono WD25R WD50R WD75R \
    --skip-etccdi \
    --skip-multi \
    --mask-type land \
    --period1-start 1980 \
    --period1-end 1999 \
    --period2-start 2000 \
    --period2-end 2024
```

#### Figure 4: Multi-Threshold Intensity vs Chronological Comparison

**Filename:** `figures/multi_threshold_intensity_chronological_comparison_land.png`  
**Script:** `visualization/ratio_difference.py`  
**Description:** Compares intensity-based (WD25/50/75) vs chronological (WD25R/50R/75R) indices, showing ratio and difference patterns.

**Command:**
```bash
python visualization/ratio_difference.py \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --mask-type land \
    --percentiles 25 50 75 \
    --threshold 0.0
```

#### Figure 5: Unified Trends (No Significance Contours)

**Filename:** `figures/unified_trends_1980-2024_land_no_sig.png`  
**Script:** `visualization/trends.py`  
**Description:** Shows Mann-Kendall trends for ETCCDI, WDXX, and WDXXR indices over 1980-2024 period without significance contours.

**Command:**
```bash
python visualization/trends.py \
    --etccdi-dir data/processed/etccdi_indices \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --mask-type land \
    --no-significance \
    --percentiles 25 50 75
```

### Supplementary Figures

#### Figure S1: Side-by-Side Precipitation Analysis

**Filename:** `figures/side_by_side_precipitation_analysis.png`  
**Script:** `visualization/baseline_precipitation.py`  
**Description:** Side-by-side comparison of precipitation percentiles and ETCCDI indices.

**Command:**
```bash
python visualization/baseline_precipitation.py \
    --etccdi-dir data/processed/etccdi_indices \
    --percentile-file data/processed/precipitation_percentiles.nc \
    --output-file figures/side_by_side_precipitation_analysis.png \
    --start-year 1980 \
    --end-year 2024 \
    --mask land \
    --dpi 300
```

#### Figure S2: ETCCDI Period Comparison

**Filename:** `figures/etccdi_period_comparison.png`  
**Script:** `visualization/median_change.py`  
**Description:** Compares median values of ETCCDI indices (PRCPTOT, R95p, R95pTOT, WD50) between two periods.

**Command:**
```bash
python visualization/median_change.py \
    --etccdi-dir data/processed/etccdi_indices \
    --output-dir figures/ \
    --include-etccdi PRCPTOT R95p R95pTOT WD50 \
    --skip-multi \
    --skip-chrono \
    --mask-type land \
    --period1-start 1980 \
    --period1-end 1999 \
    --period2-start 2000 \
    --period2-end 2024
```

#### Figure S3: Multi-Threshold Period Comparison

**Filename:** `figures/multi_threshold_period_comparison.png`  
**Script:** `visualization/median_change.py`  
**Description:** Same as Figure 2 (see command above).

**Command:**
```bash
# Same command as Figure 2
python visualization/median_change.py \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --output-dir figures/ \
    --include-multi WD25 WD50 WD75 \
    --skip-etccdi \
    --skip-chrono \
    --mask-type land \
    --period1-start 1980 \
    --period1-end 1999 \
    --period2-start 2000 \
    --period2-end 2024
```

#### Figure S4: Unified Trends (With Significance Contours)

**Filename:** `figures/unified_trends_1980-2024_land_sig_contour.png`  
**Script:** `visualization/trends.py`  
**Description:** Shows Mann-Kendall trends for ETCCDI, WDXX, and WDXXR indices with statistical significance contours (p < 0.10).

**Command:**
```bash
python visualization/trends.py \
    --etccdi-dir data/processed/etccdi_indices \
    --enhanced-dir data/processed/enhanced_concentration_indices \
    --wd50r-dir data/processed/wd50r_indices \
    --output-dir figures/ \
    --mask-type land \
    --sig-method contour \
    --percentiles 25 50 75
```

### Notes

- All visualization scripts support `--land-only`, `--ocean-only`, or `--mask-type {land,ocean,both}` arguments to control the analysis domain.
- The default output directory is `figures/` (will be created if it doesn't exist).
- Period comparison figures use default periods (1980-1999 vs 2000-2024) but can be customized with `--period1-start`, `--period1-end`, `--period2-start`, `--period2-end`.
- Trend analysis scripts support parallel processing with `--n-processes` argument (default varies by script).

## Key Features

### Novel WDXXR Indices
The WDXXR (chronological clustering) indices are a novel contribution that:
- Measure consecutive days needed for precipitation accumulation
- Capture temporal clustering patterns
- Complement standard WDXX indices
- Provide regime-specific insights

### Multi-Threshold Analysis
- WD25, WD50, WD75: Standard concentration thresholds
- WD25R, WD50R, WD75R: Chronological clustering thresholds
- Enables comprehensive precipitation pattern characterization

### High-Performance Computing
- Parallel processing support
- Configurable chunk sizes
- Memory-efficient processing
- Suitable for global analysis

## Output Files

### Combined Time-Series Files
- `wdxx_1980_2024.nc`: WD25, WD50, WD75 (standard concentration)
- `wdxxr_1980_2024.nc`: WD25R, WD50R, WD75R (chronological clustering)

### Annual Files
- Individual year files for each index type
- Organized by processing step

### Visualization Outputs
- Publication-quality figures (PNG format)
- Configurable for land/ocean/both analysis
- Manuscript-ready outputs

## Computational Requirements

- **Memory**: ~2-4 GB per process
- **Storage**: ~500 MB per year per method
- **Recommended**: 24-48 processes for optimal performance
- **Time**: ~0.5-1 hour per year depending on system

## Citation

If you use this code, please cite:

```

```

## License

This code is provided under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Saurav Bhattarai**  
Jackson State University  
Email: saurav.bhattarai@students.jsums.edu or saurav.bhattarai.1999@gmail.com

## Acknowledgments

The authors acknowledge the support of the Oak Ridge Institute for Science and Education (ORISE) fellowship program under the Department of Defense at the U.S. Army Engineer Research and Development Center (ERDC), USACE. This research is also supported by the Hydrological Impacts Computing, Outreach, and Resiliency Partnership (HICORPS) Project, developed in collaboration with the U.S. Army Engineer Research and Development Center (ERDC), WOOLPERT, and Taylor Engineering. We thank Jackson State University (JSU) for providing computational resources and infrastructure support throughout this research.

We acknowledge the European Centre for Medium-Range Weather Forecasts (ECMWF) for providing the ERA5 reanalysis dataset through the Copernicus Climate Change Service (C3S) Climate Data Store (https://cds.climate.copernicus.eu/).

## Author Contributions

**Saurav Bhattarai:** Conceptualization, methodology development, data curation, formal analysis, software implementation, visualization, writing—original draft preparation, writing—review and editing.

**Nawa Raj Pradhan:** Conceptualization, methodology guidance, supervision, writing—review and editing, funding acquisition.

**Rocky Talchabhadel:** Conceptualization, methodology development, formal analysis guidance, supervision, writing—review and editing, validation of analytical approaches.
