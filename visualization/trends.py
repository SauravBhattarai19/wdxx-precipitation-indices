#!/usr/bin/env python3
"""
Unified Trends Analysis: ETCCDI, WDXX, and WDXXR Precipitation Indices

High-performance version with:
- Vectorized Mann-Kendall trend analysis
- Multiprocessing for 192-core systems
- Permanent data storage (no caching)
- Scientific-grade statistical testing

Figure Layout: 3 rows × 3 columns (1980-2024 full period only)
- Row 1: ETCCDI indices (PRCPTOT, R95p, R95pTOT)
- Row 2: WDXX multi-threshold indices (WD25, WD50, WD75)
- Row 3: WDXXR chronological indices (WD25R, WD50R, WD75R)

Usage:
python trends.py --etccdi-dir data/processed/etccdi_indices --enhanced-dir data/processed/enhanced_concentration_indices --wd50r-dir data/processed/wd50r_indices --output-dir figures/
"""

import sys
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import pandas as pd
from scipy import stats
import warnings
import multiprocessing as mp
from multiprocessing import Pool
import time
from functools import partial
import h5py
from numba import jit, prange
import os

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    print("Warning: regionmask not available - using simplified land/ocean mask")

warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

@jit(nopython=True, parallel=True)
def mann_kendall_vectorized(data_3d, years):
    """
    Vectorized Mann-Kendall trend test using Numba for speed.
    """
    nt, nlat, nlon = data_3d.shape
    slopes = np.full((nlat, nlon), np.nan)
    p_values = np.full((nlat, nlon), np.nan)
    z_stats = np.full((nlat, nlon), np.nan)

    # Precompute year differences for Sen's slope
    year_diffs = np.zeros((nt * (nt - 1) // 2,))
    k = 0
    for i in range(nt - 1):
        for j in range(i + 1, nt):
            year_diffs[k] = years[j] - years[i]
            k += 1

    for i in prange(nlat):
        for j in range(nlon):
            # Extract time series for this pixel
            ts = data_3d[:, i, j]

            # Skip if too many NaN values
            valid_mask = ~np.isnan(ts)
            n_valid = np.sum(valid_mask)

            if n_valid < 4:  # Need at least 4 points for Mann-Kendall
                continue

            # Extract valid data and corresponding years
            valid_data = ts[valid_mask]
            valid_years = years[valid_mask]
            n = len(valid_data)

            # Mann-Kendall S statistic
            S = 0
            slopes_list = []

            # Calculate all pairwise comparisons
            for ii in range(n - 1):
                for jj in range(ii + 1, n):
                    diff = valid_data[jj] - valid_data[ii]
                    if diff > 0:
                        S += 1
                    elif diff < 0:
                        S -= 1

                    # Sen's slope calculation
                    if valid_years[jj] != valid_years[ii]:
                        slope = diff / (valid_years[jj] - valid_years[ii])
                        slopes_list.append(slope)

            # Sen's slope (median of all slopes)
            if len(slopes_list) > 0:
                slopes_array = np.array(slopes_list)
                slopes[i, j] = np.median(slopes_array)

            # Variance of S (assuming no ties for simplicity)
            var_s = n * (n - 1) * (2 * n + 5) / 18.0

            # Z-statistic
            if S > 0:
                z = (S - 1) / np.sqrt(var_s)
            elif S < 0:
                z = (S + 1) / np.sqrt(var_s)
            else:
                z = 0

            z_stats[i, j] = z

            # Two-tailed p-value (approximation using normal distribution)
            if var_s > 0:
                p_values[i, j] = 2 * (1 - 0.5 * (1 + np.sign(np.abs(z)) * np.sqrt(1 - np.exp(-2 * z * z / np.pi))))
            else:
                p_values[i, j] = 1.0

    return slopes, p_values, z_stats

def enhanced_mann_kendall_with_ties(data, years):
    """
    Enhanced Mann-Kendall test accounting for ties in data.
    """
    from scipy import stats as scipy_stats

    # Remove NaN values
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 4:
        return np.nan, np.nan, np.nan

    valid_data = data[valid_mask]
    valid_years = years[valid_mask]
    n = len(valid_data)

    # Mann-Kendall S statistic
    S = 0
    slopes_list = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = valid_data[j] - valid_data[i]
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1

            # Sen's slope
            if valid_years[j] != valid_years[i]:
                slope = diff / (valid_years[j] - valid_years[i])
                slopes_list.append(slope)

    # Sen's slope (median)
    sen_slope = np.median(slopes_list) if slopes_list else np.nan

    # Account for ties in variance calculation
    unique_vals, counts = np.unique(valid_data, return_counts=True)
    tie_adjustment = np.sum(counts * (counts - 1) * (2 * counts + 5)) / 18.0

    var_s = (n * (n - 1) * (2 * n + 5) - tie_adjustment) / 18.0

    if var_s <= 0:
        return sen_slope, 1.0, 0.0

    # Z-statistic with continuity correction
    if S > 0:
        z = (S - 1) / np.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Two-tailed p-value
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return sen_slope, p_value, z

def process_pixel_chunk(args):
    """Process a chunk of pixels for multiprocessing."""
    data_chunk, years, i_start, i_end, j_start, j_end = args

    slopes = np.full((i_end - i_start, j_end - j_start), np.nan)
    p_values = np.full((i_end - i_start, j_end - j_start), np.nan)
    z_stats = np.full((i_end - i_start, j_end - j_start), np.nan)

    for i in range(i_end - i_start):
        for j in range(j_end - j_start):
            pixel_data = data_chunk[:, i, j]
            slope, p_val, z_stat = enhanced_mann_kendall_with_ties(pixel_data, years)
            slopes[i, j] = slope
            p_values[i, j] = p_val
            z_stats[i, j] = z_stat

    return slopes, p_values, z_stats, i_start, i_end, j_start, j_end

def calculate_trends_parallel(data_array, years, n_processes=None, chunk_size=50):
    """
    Calculate Mann-Kendall trends using multiprocessing.
    """
    if n_processes is None:
        n_processes = min(192, mp.cpu_count())

    print(f"Using {n_processes} processes for parallel trend calculation")

    # Get data dimensions
    data_3d = data_array.values
    nt, nlat, nlon = data_3d.shape

    # Initialize output arrays
    slopes = np.full((nlat, nlon), np.nan)
    p_values = np.full((nlat, nlon), np.nan)
    z_stats = np.full((nlat, nlon), np.nan)

    # Create chunks for parallel processing
    chunks = []
    for i in range(0, nlat, chunk_size):
        for j in range(0, nlon, chunk_size):
            i_end = min(i + chunk_size, nlat)
            j_end = min(j + chunk_size, nlon)

            # Extract data chunk
            data_chunk = data_3d[:, i:i_end, j:j_end]
            chunks.append((data_chunk, years, i, i_end, j, j_end))

    print(f"Processing {len(chunks)} spatial chunks")

    # Process chunks in parallel
    with Pool(n_processes) as pool:
        results = pool.map(process_pixel_chunk, chunks)

    # Combine results
    for chunk_slopes, chunk_p_values, chunk_z_stats, i_start, i_end, j_start, j_end in results:
        slopes[i_start:i_end, j_start:j_end] = chunk_slopes
        p_values[i_start:i_end, j_start:j_end] = chunk_p_values
        z_stats[i_start:i_end, j_start:j_end] = chunk_z_stats

    return slopes, p_values, z_stats

def create_optimized_land_ocean_mask(ds, mask_type='both', validate=True):
    """
    Optimized land/ocean mask creation with validation.
    """
    if mask_type == 'both':
        return np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)

    lat = ds.latitude.values
    lon = ds.longitude.values

    print(f"Creating optimized {mask_type} mask for {len(lat)}x{len(lon)} grid...")

    if HAS_REGIONMASK:
        try:
            # Use regionmask with proper coordinate handling
            dummy_ds = xr.Dataset(coords={'lat': lat, 'lon': lon})

            try:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(dummy_ds)
                print("  Using 50m resolution Natural Earth boundaries")
            except:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
                print("  Using 110m resolution Natural Earth boundaries")

            land_mask = ~np.isnan(land_mask_rm.values)

            # Clean up mask
            from scipy.ndimage import binary_opening, binary_closing
            if land_mask.sum() > 0:
                land_mask = binary_closing(land_mask, iterations=1)
                land_mask = binary_opening(land_mask, iterations=1)

            if validate:
                land_pct = np.sum(land_mask) / land_mask.size * 100
                ocean_pct = 100 - land_pct
                print(f"  Land pixels: {np.sum(land_mask):,} ({land_pct:.1f}%)")
                print(f"  Ocean pixels: {np.sum(~land_mask):,} ({ocean_pct:.1f}%)")

                # Validate reasonable land/ocean ratio
                if land_pct < 25 or land_pct > 40:
                    print(f"  Warning: Unusual land percentage ({land_pct:.1f}%) - check mask quality")

            final_mask = land_mask if mask_type == 'land' else ~land_mask

            if validate:
                print(f"  Final {mask_type} mask: {np.sum(final_mask):,} pixels ({np.sum(final_mask)/final_mask.size*100:.1f}%)")

            return final_mask

        except Exception as e:
            print(f"  regionmask failed: {e}, using fallback")

    # Enhanced fallback approach
    print("  Using enhanced geometric land/ocean approximation...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    land_mask = np.zeros_like(lat_grid, dtype=bool)

    land_regions = [
        # North America
        ((lat_grid >= 25) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -50)),
        # South America
        ((lat_grid >= -60) & (lat_grid <= 15) & (lon_grid >= -85) & (lon_grid <= -30)),
        # Europe
        ((lat_grid >= 35) & (lat_grid <= 75) & (lon_grid >= -15) & (lon_grid <= 45)),
        # Asia
        ((lat_grid >= 5) & (lat_grid <= 75) & (lon_grid >= 25) & (lon_grid <= 180)),
        # Africa
        ((lat_grid >= -35) & (lat_grid <= 40) & (lon_grid >= -20) & (lon_grid <= 55)),
        # Australia
        ((lat_grid >= -50) & (lat_grid <= -5) & (lon_grid >= 110) & (lon_grid <= 180)),
        # Greenland
        ((lat_grid >= 60) & (lat_grid <= 85) & (lon_grid >= -75) & (lon_grid <= -10)),
        # Antarctica
        (lat_grid <= -60),
    ]

    for region in land_regions:
        land_mask |= region

    if validate:
        land_pct = np.sum(land_mask) / land_mask.size * 100
        print(f"  Fallback land percentage: {land_pct:.1f}%")

    return land_mask if mask_type == 'land' else ~land_mask

def load_etccdi_data_optimized(etccdi_dir, years=None):
    """
    Optimized ETCCDI data loading with memory-efficient operations.
    """
    etccdi_dir = Path(etccdi_dir)

    print(f"Searching for ETCCDI files in: {etccdi_dir.absolute()}")

    if years is None:
        # Check if directory exists
        if not etccdi_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {etccdi_dir}")

        # Look for ETCCDI files with various possible patterns
        patterns = [
            'etccdi_precipitation_indices_*.nc',
            'etccdi_*.nc',
            '*precipitation*indices*.nc',
            '*etccdi*.nc',
            '*.nc'  # fallback to show all .nc files
        ]

        etccdi_files = []
        all_nc_files = []

        for pattern in patterns:
            files = list(etccdi_dir.glob(pattern))
            all_nc_files.extend([f for f in files if f not in all_nc_files])
            if pattern == 'etccdi_precipitation_indices_*.nc' and files:
                etccdi_files = files
                break
            elif pattern == 'etccdi_*.nc' and files and not etccdi_files:
                etccdi_files = files
            elif pattern == '*precipitation*indices*.nc' and files and not etccdi_files:
                etccdi_files = files
            elif pattern == '*etccdi*.nc' and files and not etccdi_files:
                etccdi_files = files

        print(f"Found {len(all_nc_files)} .nc files total:")
        for f in sorted(all_nc_files)[:10]:  # Show first 10
            print(f"  {f.name}")
        if len(all_nc_files) > 10:
            print(f"  ... and {len(all_nc_files) - 10} more")

        if not etccdi_files:
            raise FileNotFoundError(
                f"No ETCCDI precipitation index files found in {etccdi_dir}\n"
                f"Expected pattern: 'etccdi_precipitation_indices_YYYY.nc'\n"
                f"Please check your file naming convention or directory path."
            )

        print(f"Using {len(etccdi_files)} ETCCDI files:")
        for f in sorted(etccdi_files)[:5]:  # Show first 5
            print(f"  {f.name}")
        if len(etccdi_files) > 5:
            print(f"  ... and {len(etccdi_files) - 5} more")

        # Extract years from filenames
        try:
            years = []
            for f in etccdi_files:
                # Try different filename patterns to extract year
                name_parts = f.stem.split('_')
                year_found = False

                # Look for 4-digit years in filename parts
                for part in name_parts:
                    if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2030:
                        years.append(int(part))
                        year_found = True
                        break

                if not year_found:
                    print(f"Warning: Could not extract year from {f.name}")

            if not years:
                raise ValueError("Could not extract years from any filenames")

            years = sorted(set(years))
            print(f"Extracted years: {years[0]}-{years[-1]} ({len(years)} years)")

        except (ValueError, IndexError) as e:
            raise ValueError(f"Could not extract years from filenames: {e}")

    print(f"Loading ETCCDI data for {len(years)} years...")

    # Load first file to get dimensions and variables
    first_file = etccdi_dir / f'etccdi_precipitation_indices_{years[0]}.nc'
    with xr.open_dataset(first_file) as ds_first:
        variables = list(ds_first.data_vars)
        lat_coords = ds_first.latitude.values
        lon_coords = ds_first.longitude.values

    print(f"Variables: {variables}")
    print(f"Grid: {len(lat_coords)} x {len(lon_coords)}")

    # Pre-allocate arrays for efficient loading
    data_arrays = {}
    for var in ['PRCPTOT', 'R95p', 'R95pTOT']:
        if var in variables:
            data_arrays[var] = np.full((len(years), len(lat_coords), len(lon_coords)), np.nan)

    # Load data year by year
    valid_years = []
    for i, year in enumerate(years):
        file_path = etccdi_dir / f'etccdi_precipitation_indices_{year}.nc'

        if file_path.exists():
            try:
                with xr.open_dataset(file_path) as ds:
                    # Process timedelta variables
                    timedelta_vars = ['WD50', 'wet_days', 'very_wet_days']
                    for var in timedelta_vars:
                        if var in ds.data_vars and 'timedelta' in str(ds[var].dtype):
                            ds[var] = ds[var] / np.timedelta64(1, 'D')
                            ds[var] = ds[var].astype(np.float64)

                    # Store data in pre-allocated arrays
                    for var in data_arrays.keys():
                        if var in ds.data_vars:
                            data_arrays[var][i] = ds[var].values

                valid_years.append(year)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

    # Create optimized xarray dataset
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }

    data_vars = {}
    for var, data in data_arrays.items():
        # Only keep years that loaded successfully
        data_vars[var] = xr.DataArray(
            data[:len(valid_years)],
            dims=['year', 'latitude', 'longitude'],
            coords=coords
        )

    combined_ds = xr.Dataset(data_vars, coords=coords)

    print(f"Successfully loaded: {len(valid_years)} years, {len(data_vars)} variables")
    return combined_ds

def load_enhanced_concentration_data_optimized(enhanced_dir, years=None):
    """
    Load WD25, WD50, WD75 data from enhanced concentration indices.
    """
    enhanced_dir = Path(enhanced_dir)

    print(f"Loading enhanced concentration data from: {enhanced_dir.absolute()}")

    if years is None:
        # Find enhanced concentration files
        enhanced_files = list(enhanced_dir.glob('enhanced_concentration_indices_*.nc'))

        if not enhanced_files:
            raise FileNotFoundError(f"No enhanced concentration files found in {enhanced_dir}")

        print(f"Found {len(enhanced_files)} enhanced concentration files")

        # Extract years
        years = []
        for f in enhanced_files:
            name_parts = f.stem.split('_')
            for part in name_parts:
                if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2030:
                    years.append(int(part))
                    break

        years = sorted(set(years))
        print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")

    # Load first file to get dimensions and variables
    first_file = None
    for f in enhanced_files:
        if any(str(year) in f.name for year in years[:5]):  # Try first few years
            first_file = f
            break

    if first_file is None:
        first_file = enhanced_files[0]

    with xr.open_dataset(first_file) as ds_first:
        variables = list(ds_first.data_vars)
        lat_coords = ds_first.latitude.values
        lon_coords = ds_first.longitude.values

    print(f"Available variables: {variables}")
    print(f"Grid: {len(lat_coords)} x {len(lon_coords)}")

    # Focus on WD25, WD50, WD75
    target_vars = ['WD25', 'WD50', 'WD75']
    available_vars = [var for var in target_vars if var in variables]

    if not available_vars:
        raise ValueError(f"None of the target variables {target_vars} found in files. Available: {variables}")

    print(f"Target variables: {available_vars}")

    # Pre-allocate arrays for efficient loading
    data_arrays = {}
    for var in available_vars:
        data_arrays[var] = np.full((len(years), len(lat_coords), len(lon_coords)), np.nan)

    # Load data year by year
    valid_years = []
    for i, year in enumerate(years):
        # Find files for this year (try different patterns)
        year_files = list(enhanced_dir.glob(f'enhanced_concentration_indices_{year}_*.nc'))

        if not year_files:
            # Try alternative pattern
            year_files = [f for f in enhanced_files if f'_{year}_' in f.name or f.name.endswith(f'_{year}.nc')]

        if year_files:
            file_path = year_files[0]  # Take first match

            try:
                with xr.open_dataset(file_path) as ds:
                    # Store data in pre-allocated arrays
                    for var in available_vars:
                        if var in ds.data_vars:
                            var_data = ds[var].values

                            # Handle potential timedelta conversion
                            if hasattr(var_data, 'dtype') and 'timedelta' in str(var_data.dtype):
                                print(f"  Converting {var} from timedelta to days")
                                var_data = var_data / np.timedelta64(1, 'D')
                                var_data = var_data.astype(np.float64)

                            data_arrays[var][i] = var_data

                valid_years.append(year)
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        else:
            print(f"  Warning: No file found for year {year}")

    if not valid_years:
        raise ValueError("No valid years loaded")

    # Create optimized xarray dataset
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }

    data_vars = {}
    for var, data in data_arrays.items():
        # Only keep years that loaded successfully
        data_vars[var] = xr.DataArray(
            data[:len(valid_years)],
            dims=['year', 'latitude', 'longitude'],
            coords=coords
        )

    combined_ds = xr.Dataset(data_vars, coords=coords)

    print(f"Successfully loaded: {len(valid_years)} years, {len(data_vars)} variables")
    return combined_ds

def load_wdxxr_data_optimized(wd50r_dir, percentiles=[25, 50, 75], years=None):
    """
    Load WDXXR data for multiple percentiles from chronological precipitation indices.
    """
    wd50r_dir = Path(wd50r_dir)

    print(f"Loading WDXXR data from: {wd50r_dir.absolute()}")
    print(f"Target percentiles: {percentiles}")

    if years is None:
        # Find WD50R files (assuming they contain all percentiles)
        wd50r_files = list(wd50r_dir.glob('wd50r_indices_*.nc'))

        if not wd50r_files:
            raise FileNotFoundError(f"No WDXXR files found in {wd50r_dir}")

        print(f"Found {len(wd50r_files)} WDXXR files")

        # Extract years
        years = []
        for f in wd50r_files:
            name_parts = f.stem.split('_')
            for part in name_parts:
                if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2030:
                    years.append(int(part))
                    break

        years = sorted(set(years))
        print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")

    # Load WDXXR data for all percentiles
    wdxxr_datasets = {}

    for percentile in percentiles:
        print(f"\nLoading WD{percentile}R data...")
        wdxxr_data_list = []
        valid_years = []

        for year in years:
            # Find file for this year
            year_files = list(wd50r_dir.glob(f'wd50r_indices_{year}_*.nc'))

            if year_files:
                file_path = year_files[0]  # Take first match

                try:
                    with xr.open_dataset(file_path) as ds:
                        var_name = f'WD{percentile}R'

                        if var_name in ds.data_vars:
                            wd_var = ds[var_name]

                            # Get coordinates from first file
                            if len(valid_years) == 1:
                                lat_coords = ds.latitude.values
                                lon_coords = ds.longitude.values

                            # Handle timedelta conversion
                            if 'timedelta' in str(wd_var.dtype):
                                wd_values = wd_var / np.timedelta64(1, 'D')
                                wd_values = wd_values.astype(np.float64)
                            elif 'datetime' in str(wd_var.dtype):
                                wd_values = wd_var.astype(np.float64)
                            else:
                                # Check for suspiciously large values
                                sample_vals = wd_var.values.flatten()
                                valid_sample = sample_vals[~np.isnan(sample_vals)]

                                if len(valid_sample) > 0 and np.mean(valid_sample) > 1e10:
                                    print(f"  Detected nanoseconds, converting WD{percentile}R to days")
                                    wd_values = wd_var / 1e9 / 86400  # ns -> days
                                    wd_values = wd_values.astype(np.float64)
                                elif len(valid_sample) > 0 and np.mean(valid_sample) > 1e6:
                                    print(f"  Detected microseconds, converting WD{percentile}R to days")
                                    wd_values = wd_var / 1e6 / 86400  # μs -> days
                                    wd_values = wd_values.astype(np.float64)
                                else:
                                    wd_values = wd_var.astype(np.float64)

                            wdxxr_data_list.append(wd_values.values)
                            valid_years.append(year)

                        else:
                            print(f"  Warning: WD{percentile}R not found in {file_path}")

                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")

        if not wdxxr_data_list:
            print(f"  Warning: No valid WD{percentile}R data found, skipping")
            continue

        # Combine into dataset for this percentile
        wdxxr_array = np.stack(wdxxr_data_list, axis=0)

        coords = {
            'year': valid_years,
            'latitude': lat_coords,
            'longitude': lon_coords
        }

        wdxxr_da = xr.DataArray(
            wdxxr_array,
            dims=['year', 'latitude', 'longitude'],
            coords=coords,
            name=f'WD{percentile}R'
        )

        wdxxr_datasets[f'WD{percentile}R'] = wdxxr_da

        # Validation
        sample_data = wdxxr_da.values.flatten()
        valid_final = sample_data[~np.isnan(sample_data)]

        if len(valid_final) > 0:
            print(f"  Successfully loaded WD{percentile}R: {len(valid_years)} years, {wdxxr_array.shape[1]}x{wdxxr_array.shape[2]} grid")
            print(f"  Data range: {np.min(valid_final):.2f} to {np.max(valid_final):.2f} days")
            print(f"  Data mean: {np.mean(valid_final):.2f} ± {np.std(valid_final):.2f} days")

    if not wdxxr_datasets:
        raise ValueError("No valid WDXXR data found for any percentile")

    # Combine all percentiles into single dataset
    combined_ds = xr.Dataset(wdxxr_datasets, coords=coords)

    print(f"\nSuccessfully loaded {len(wdxxr_datasets)} WDXXR variables")
    return combined_ds

def add_significance_contours(ax, ds, significant, mask, line_width=0.8, alpha=0.9):
    """
    Optimized significance visualization using contours with proper masking.
    """
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return

    try:
        # Create a float array for contouring, ensuring masked areas are handled
        sig_float = np.zeros_like(significant, dtype=float)
        sig_float[significant_masked] = 1.0
        sig_float[~mask] = 0.0  # Explicitly set non-mask areas to 0

        # Handle longitude coordinates for cartopy
        lon_vals = ds.longitude.values
        if lon_vals[0] > 180:  # Convert 0-360 to -180-180 if needed
            plot_lon = lon_vals - 360
        else:
            plot_lon = lon_vals

        contours = ax.contour(plot_lon, ds.latitude, sig_float,
                            levels=[0.5], colors='black', linewidths=line_width,
                            transform=ccrs.PlateCarree(), alpha=alpha)
        return contours
    except Exception as e:
        print(f"Warning: Could not draw significance contours: {e}")
        return None

def create_unified_trends_figure(etccdi_ds, wdxx_ds, wdxxr_ds, output_dir, mask_type='both',
                                show_significance=True, significance_method='contour'):
    """
    Create unified trends figure (3 rows × 3 columns) for full period (1980-2024).
    """
    # Create masks for each dataset separately to handle coordinate differences
    mask_etccdi = create_optimized_land_ocean_mask(etccdi_ds, mask_type)
    mask_wdxx = create_optimized_land_ocean_mask(wdxx_ds, mask_type)
    mask_wdxxr = create_optimized_land_ocean_mask(wdxxr_ds, mask_type)

    # Define index configurations for each row
    row_configs = {
        0: {  # Row 1: ETCCDI indices
            'ds': etccdi_ds,
            'mask': mask_etccdi,
            'indices': ['PRCPTOT', 'R95p', 'R95pTOT'],
            'units': ['mm/decade', 'mm/decade', 'fraction/decade'],
            'cmaps': ['BrBG', 'BrBG', 'RdBu_r'],
            'symmetric': [True, True, True],
            'titles': ['PRCPTOT', 'R95p', 'R95pTOT']
        },
        1: {  # Row 2: WDXX multi-threshold indices
            'ds': wdxx_ds,
            'mask': mask_wdxx,
            'indices': ['WD25', 'WD50', 'WD75'],
            'units': ['days/decade', 'days/decade', 'days/decade'],
            'cmaps': ['RdYlBu', 'RdYlBu', 'RdYlBu'],
            'symmetric': [True, True, True],
            'titles': ['WD25', 'WD50', 'WD75']
        },
        2: {  # Row 3: WDXXR chronological indices
            'ds': wdxxr_ds,
            'mask': mask_wdxxr,
            'indices': ['WD25R', 'WD50R', 'WD75R'],
            'units': ['days/decade', 'days/decade', 'days/decade'],
            'cmaps': ['RdYlBu', 'RdYlBu', 'RdYlBu'],
            'symmetric': [True, True, True],
            'titles': ['WD25R', 'WD50R', 'WD75R']
        }
    }

    # Period configuration (full period only)
    period_name = '1980-2024'
    start_year, end_year = 1980, 2024

    # Create figure with space for horizontal colorbars
    fig = plt.figure(figsize=(13, 13))  # Wider and taller for horizontal colorbars
    proj = ccrs.Robinson(central_longitude=0)
    
    # Set layout FIRST before creating any subplots or colorbars
    # This ensures consistent positioning throughout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.12,
                       wspace=0.05, hspace=0.25)  # Increased hspace for colorbar room

    subplot_idx = 0
    
    # Store subplot axes and data for colorbar creation AFTER all subplots are made
    subplot_data = []

    for row_idx, row_config in row_configs.items():
        ds = row_config['ds']
        mask = row_config['mask']
        indices = row_config['indices']
        units = row_config['units']
        cmaps = row_config['cmaps']
        symmetric_flags = row_config['symmetric']
        titles = row_config['titles']

        print(f"\nProcessing Row {row_idx + 1}: {titles}")

        # Plot each column in this row
        for col_idx, index_name in enumerate(indices):
            if index_name not in ds.data_vars:
                print(f"Warning: {index_name} not found in dataset")
                continue

            print(f"  Calculating trends for {index_name}...")

            # Get period data
            period_ds = ds.sel(year=slice(start_year, end_year))
            period_years = period_ds.year.values.astype(int)

            if len(period_years) < 4:
                print(f"  Skipping {period_name}: insufficient data ({len(period_years)} years)")
                continue

            # Calculate trends
            slopes, p_values, z_stats = calculate_trends_parallel(
                period_ds[index_name], period_years, n_processes=128
            )

            significant = p_values < 0.10

            # Apply mask
            slopes_masked = slopes.copy()
            slopes_masked[~mask] = np.nan

            # Convert to per-decade (multiply by 10) - same as original codes
            slopes_masked = slopes_masked * 10

            # Determine colorbar range using same method as original codes
            valid_data = slopes_masked[mask & ~np.isnan(slopes_masked)]
            if len(valid_data) == 0:
                print(f"  Warning: No valid data for {index_name} after masking")
                continue

            # Calculate actual data range for colorbar label
            actual_min = np.min(valid_data)
            actual_max = np.max(valid_data)
            
            # Special handling for R95pTOT to handle outliers better
            if titles[col_idx] == 'R95pTOT':
                # Use percentiles for R95pTOT to handle outliers
                vmin = np.percentile(valid_data, 5)  # Use 5th percentile for more contrast
                vmax = np.percentile(valid_data, 95)  # Use 95th percentile
                percentile_range = "5–95"
            elif symmetric_flags[col_idx]:
                # Use 98th percentile like original wdxx_trends.py
                vmax_abs = np.percentile(np.abs(valid_data), 98)
                vmin, vmax = -vmax_abs, vmax_abs
                percentile_range = "2–98"
            else:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
                percentile_range = "2–98"

            levels = np.linspace(vmin, vmax, 21)

            subplot_idx += 1

            # Create subplot with space for horizontal colorbar below
            ax = plt.subplot(3, 3, subplot_idx, projection=proj)

            # Map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
            ax.set_global()

            # Plot trends using pcolormesh to prevent color leakage
            cmap = plt.cm.get_cmap(cmaps[col_idx])
            # slopes_masked is already converted to per-decade above

            # Use pcolormesh with flat shading to prevent interpolation artifacts
            lon_vals = ds.longitude.values
            lat_vals = ds.latitude.values

            # Create coordinate arrays for pcolormesh (need to be 1 element larger than data)
            if len(lon_vals) > 1:
                lon_spacing = lon_vals[1] - lon_vals[0]
                lon_bounds = np.concatenate([lon_vals - lon_spacing/2,
                                           [lon_vals[-1] + lon_spacing/2]])
            else:
                lon_bounds = np.array([lon_vals[0] - 0.5, lon_vals[0] + 0.5])

            if len(lat_vals) > 1:
                lat_spacing = lat_vals[1] - lat_vals[0]
                lat_bounds = np.concatenate([lat_vals - lat_spacing/2,
                                           [lat_vals[-1] + lat_spacing/2]])
            else:
                lat_bounds = np.array([lat_vals[0] - 0.5, lat_vals[0] + 0.5])

            # Create masked array to ensure no color leakage
            slopes_ma = np.ma.masked_invalid(slopes_masked)
            slopes_ma = np.ma.masked_where(~mask, slopes_ma)  # Double mask

            # Ensure longitude coordinates are properly handled for cartopy
            if lon_vals[0] > 180:  # Convert 0-360 to -180-180 if needed
                lon_bounds = lon_bounds - 360

            im = ax.pcolormesh(lon_bounds, lat_bounds, slopes_ma,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree(), shading='flat')

            # Add significance
            if show_significance:
                if significance_method == 'contour':
                    add_significance_contours(ax, ds, significant, mask)

            # Calculate and display statistics (like wdxx_trends.py)
            if show_significance:
                # Total significant pixels
                sig_pixels = np.sum(significant & mask)
                total_pixels = np.sum(mask)
                sig_percentage = sig_pixels / total_pixels * 100 if total_pixels > 0 else 0

                # Separate increasing and decreasing trends
                increasing_sig = np.sum((significant & mask) & (slopes > 0))
                decreasing_sig = np.sum((significant & mask) & (slopes < 0))

                increasing_pct = increasing_sig / total_pixels * 100 if total_pixels > 0 else 0
                decreasing_pct = decreasing_sig / total_pixels * 100 if total_pixels > 0 else 0

                # Display statistics in bottom-left corner
                stats_text = f'{sig_percentage:.1f}% sig.\n+{increasing_pct:.1f}% ↑\n-{decreasing_pct:.1f}% ↓'
                ax.text(0.02, 0.02, stats_text,
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       verticalalignment='bottom', fontweight='bold')

            # Add title with index name
            label = chr(97 + subplot_idx - 1)  # a, b, c, d, e, f, g, h, i
            title_text = f"({label}) {titles[col_idx]}"
            ax.set_title(title_text, fontsize=14, fontweight='bold', pad=5)

            # Store data for colorbar creation after all subplots are made
            subplot_data.append({
                'ax': ax,
                'cmap': cmap,
                'vmin': vmin,
                'vmax': vmax,
                'units': units[col_idx],
                'percentile_range': percentile_range,
                'actual_min': actual_min,
                'actual_max': actual_max
            })

    # Now create colorbars using the FINAL adjusted subplot positions
    print(f"\nCreating {len(subplot_data)} colorbars...")
    for data in subplot_data:
        ax = data['ax']
        
        # Get the FINAL position after subplots_adjust
        ax_pos = ax.get_position()
        
        # Calculate colorbar position with proper alignment
        cbar_width = ax_pos.width
        cbar_height = 0.020  # Slightly smaller for cleaner look
        cbar_x = ax_pos.x0
        cbar_y = ax_pos.y0 - 0.055  # Position below subplot with consistent gap
        
        # Create colorbar axes with figure coordinates
        cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        
        # Create normalized colorbar with extend='both' for extremes
        norm = mcolors.Normalize(vmin=data['vmin'], vmax=data['vmax'])
        dummy_im = plt.cm.ScalarMappable(norm=norm, cmap=data['cmap'])
        cbar = fig.colorbar(dummy_im, cax=cbar_ax, orientation='horizontal', extend='both')
        
        # Comprehensive colorbar label with units, percentile range, and actual data range
        cbar_label = f"{data['units']}\nColor scale: {data['percentile_range']} percentile\nActual range: {data['actual_min']:.2f} to {data['actual_max']:.2f}"
        cbar.set_label(cbar_label, fontsize=8, linespacing=1.2)
        
        # Set ticks to be strictly within [vmin, vmax]
        tick_locs = np.linspace(data['vmin'], data['vmax'], 5)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([f'{val:.2f}' for val in tick_locs])
        cbar.ax.tick_params(labelsize=7)
    
    # Add significance note as a text annotation if needed
    if show_significance:
        fig.text(0.02, 0.02, 'Black contours indicate statistical significance (p < 0.10)',
                fontsize=12, style='italic', ha='left')

    # Save
    filename = f'unified_trends_1980-2024_{mask_type}'
    if show_significance:
        filename += f'_sig_{significance_method}'
    else:
        filename += '_no_sig'
    filename += '.png'

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure saved: {output_path}")
    return output_path

def main():
    """Main function for unified trends analysis."""
    parser = argparse.ArgumentParser(description='Unified precipitation trends analysis (ETCCDI, WDXX, WDXXR)')

    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI index files')
    parser.add_argument('--enhanced-dir', default='data/processed/enhanced_concentration_indices',
                       help='Directory containing WDXX enhanced concentration index files')
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices',
                       help='Directory containing WDXXR chronological index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain (default: both)')
    parser.add_argument('--no-significance', action='store_true',
                       help='Hide statistical significance')
    parser.add_argument('--sig-method', choices=['contour'], default='contour',
                       help='Significance visualization method (default: contour)')
    parser.add_argument('--n-processes', type=int, default=128,
                       help='Number of processes for parallel computation (default: 128)')
    parser.add_argument('--land-only', action='store_true',
                       help='Shortcut for --mask-type land')
    parser.add_argument('--ocean-only', action='store_true',
                       help='Shortcut for --mask-type ocean')
    parser.add_argument('--percentiles', nargs='+', type=int, default=[25, 50, 75],
                       help='WDXXR percentiles to analyze (default: 25 50 75)')

    args = parser.parse_args()

    # Handle shortcuts
    if args.land_only:
        args.mask_type = 'land'
    elif args.ocean_only:
        args.mask_type = 'ocean'

    print("="*80)
    print("UNIFIED PRECIPITATION TRENDS ANALYSIS")
    print("="*80)
    print(f"ETCCDI directory: {args.etccdi_dir}")
    print(f"WDXX directory: {args.enhanced_dir}")
    print(f"WDXXR directory: {args.wd50r_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Show significance: {not args.no_significance}")
    print(f"Significance method: {args.sig_method}")
    print(f"Processes: {args.n_processes}")
    print(f"WDXXR percentiles: {args.percentiles}")
    print("="*80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load all datasets
        print("Loading ETCCDI data...")
        etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)

        print("\nLoading WDXX data...")
        wdxx_ds = load_enhanced_concentration_data_optimized(args.enhanced_dir)

        print("\nLoading WDXXR data...")
        wdxxr_ds = load_wdxxr_data_optimized(args.wd50r_dir, args.percentiles)

        # Create unified figure
        print("\nCreating unified trends figure...")
        output_path = create_unified_trends_figure(
            etccdi_ds, wdxx_ds, wdxxr_ds, output_dir, args.mask_type,
            show_significance=not args.no_significance,
            significance_method=args.sig_method
        )

        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print("Figure layout:")
        print("  Row 1: ETCCDI indices (PRCPTOT, R95p, R95pTOT)")
        print("  Row 2: WDXX multi-threshold indices (WD25, WD50, WD75)")
        print("  Row 3: WDXXR chronological indices (WD25R, WD50R, WD75R)")
        print("Optimizations applied:")
        print("- Vectorized Mann-Kendall trend analysis")
        print("- Multiprocessing for parallel computation")
        print("- Scientific-grade statistical testing")
        print("- Subplot labels (a-i) and proper colorbars")

        # Close datasets
        etccdi_ds.close()
        wdxx_ds.close()
        wdxxr_ds.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
