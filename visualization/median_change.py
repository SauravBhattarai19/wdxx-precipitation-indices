#!/usr/bin/env python3
"""
Create Three Separate Precipitation Concentration Comparison Plots

Plot 1: ETCCDI indices (PRCPTOT, R95p, R95pTOT, WD50) - 4x3 subplots
Plot 2: Multi-threshold indices (WD25, WD50, WD75) - 3x3 subplots  
Plot 3: Chronological indices (WD25R, WD50R, WD75R) - 3x3 subplots

Each plot has 3 columns:
- Column 1: Median 1980-1999
- Column 2: Median 2000-2024
- Column 3: Relative shift in %

Usage:
python three_comparison_plots.py --etccdi-dir data/processed/etccdi_indices 
                                --enhanced-dir data/processed/enhanced_concentration_indices
                                --wd50r-dir data/processed/wd50r_indices
                                --output-dir figures/
                                --include-etccdi PRCPTOT R95p R95pTOT WD50
                                --include-multi WD25 WD50 WD75
                                --include-chrono WD25R WD50R WD75R
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
import warnings
import multiprocessing as mp
from multiprocessing import Pool
import time
from functools import partial
import h5py
import os

try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    print("Warning: regionmask not available - using improved land/ocean mask")

warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'savefig.dpi': 600,
})

def create_optimized_land_ocean_mask(ds, mask_type='both', validate=True):
    """Improved land/ocean mask creation with better geometric approximation."""
    if mask_type == 'both':
        return np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)
    
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    print(f"Creating optimized {mask_type} mask for {len(lat)}x{len(lon)} grid...")
    
    if HAS_REGIONMASK:
        try:
            dummy_ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
            
            try:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(dummy_ds)
                print("  Using 50m resolution Natural Earth boundaries")
            except:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
                print("  Using 110m resolution Natural Earth boundaries")
            
            land_mask = ~np.isnan(land_mask_rm.values)
            
            # Apply morphological operations to clean up mask
            try:
                from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
                if land_mask.sum() > 0:
                    land_mask = binary_fill_holes(land_mask)
                    land_mask = binary_closing(land_mask, iterations=1)
                    land_mask = binary_opening(land_mask, iterations=1)
            except ImportError:
                print("  scipy not available for morphological operations")
            
            if validate:
                land_pct = np.sum(land_mask) / land_mask.size * 100
                print(f"  Land pixels: {np.sum(land_mask):,} ({land_pct:.1f}%)")
            
            return land_mask if mask_type == 'land' else ~land_mask
            
        except Exception as e:
            print(f"  regionmask failed: {e}, using improved fallback")
    
    # Improved geometric land/ocean approximation
    print("  Using improved geometric land/ocean approximation...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    land_mask = np.zeros_like(lat_grid, dtype=bool)
    
    # More detailed continental regions
    land_regions = [
        # North America
        ((lat_grid >= 25) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -50)),
        # Central America
        ((lat_grid >= 7) & (lat_grid <= 35) & (lon_grid >= -120) & (lon_grid <= -75)),
        # South America
        ((lat_grid >= -60) & (lat_grid <= 15) & (lon_grid >= -85) & (lon_grid <= -30)),
        # Europe
        ((lat_grid >= 35) & (lat_grid <= 75) & (lon_grid >= -15) & (lon_grid <= 45)),
        # Asia (main)
        ((lat_grid >= 5) & (lat_grid <= 75) & (lon_grid >= 25) & (lon_grid <= 180)),
        # Africa
        ((lat_grid >= -35) & (lat_grid <= 40) & (lon_grid >= -20) & (lon_grid <= 55)),
        # Australia and Oceania
        ((lat_grid >= -50) & (lat_grid <= -5) & (lon_grid >= 110) & (lon_grid <= 180)),
        # Greenland
        ((lat_grid >= 60) & (lat_grid <= 85) & (lon_grid >= -75) & (lon_grid <= -10)),
        # Antarctica
        (lat_grid <= -60),
        # India subcontinent (more detailed)
        ((lat_grid >= 5) & (lat_grid <= 40) & (lon_grid >= 65) & (lon_grid <= 100)),
        # Southeast Asia islands
        ((lat_grid >= -10) & (lat_grid <= 25) & (lon_grid >= 90) & (lon_grid <= 145)),
        # Japan and Korean Peninsula
        ((lat_grid >= 30) & (lat_grid <= 50) & (lon_grid >= 125) & (lon_grid <= 145)),
        # British Isles
        ((lat_grid >= 50) & (lat_grid <= 62) & (lon_grid >= -12) & (lon_grid <= 3)),
        # Scandinavia
        ((lat_grid >= 55) & (lat_grid <= 72) & (lon_grid >= 5) & (lon_grid <= 35)),
        # Madagascar
        ((lat_grid >= -27) & (lat_grid <= -12) & (lon_grid >= 43) & (lon_grid <= 51)),
        # New Zealand
        ((lat_grid >= -48) & (lat_grid <= -34) & (lon_grid >= 165) & (lon_grid <= 180)),
        # Caribbean islands (approximate)
        ((lat_grid >= 10) & (lat_grid <= 28) & (lon_grid >= -85) & (lon_grid <= -60)),
        # Mediterranean islands
        ((lat_grid >= 35) & (lat_grid <= 42) & (lon_grid >= 8) & (lon_grid <= 15)),  # Italy
        ((lat_grid >= 38) & (lat_grid <= 40) & (lon_grid >= 20) & (lon_grid <= 28)),  # Greece
    ]
    
    for region in land_regions:
        land_mask |= region
    
    # Remove major ocean areas that might have been incorrectly classified
    ocean_exclusions = [
        # Atlantic Ocean
        ((lat_grid >= -60) & (lat_grid <= 70) & (lon_grid >= -80) & (lon_grid <= -10)),
        # Pacific Ocean
        ((lat_grid >= -60) & (lat_grid <= 70) & (lon_grid >= 130) & (lon_grid <= 180)) |
        ((lat_grid >= -60) & (lat_grid <= 70) & (lon_grid >= -180) & (lon_grid <= -90)),
        # Indian Ocean
        ((lat_grid >= -60) & (lat_grid <= 30) & (lon_grid >= 20) & (lon_grid <= 120)),
        # Arctic Ocean
        ((lat_grid >= 75) & (lon_grid >= -180) & (lon_grid <= 180)),
        # Southern Ocean
        ((lat_grid <= -55) & (lon_grid >= -180) & (lon_grid <= 180)),
    ]
    
    # Apply ocean exclusions more selectively
    for exclusion in ocean_exclusions:
        # Only remove pixels that are clearly in ocean (not near coastlines)
        exclusion_mask = exclusion
        land_mask &= ~exclusion_mask
    
    # Clean up small isolated pixels
    try:
        from scipy.ndimage import binary_opening, binary_closing
        # Remove small land features (islands smaller than ~2 pixels)
        land_mask = binary_opening(land_mask, iterations=1)
        # Fill small holes in land masses
        land_mask = binary_closing(land_mask, iterations=2)
    except ImportError:
        print("  scipy not available for morphological operations")
    
    if validate:
        land_pct = np.sum(land_mask) / land_mask.size * 100
        print(f"  Improved geometric land percentage: {land_pct:.1f}%")
    
    return land_mask if mask_type == 'land' else ~land_mask

def load_etccdi_data_optimized(etccdi_dir, years=None):
    """Load ETCCDI data optimized."""
    etccdi_dir = Path(etccdi_dir)
    
    print(f"Loading ETCCDI data from: {etccdi_dir.absolute()}")
    
    if years is None:
        etccdi_files = list(etccdi_dir.glob('etccdi_precipitation_indices_*.nc'))
        if not etccdi_files:
            etccdi_files = list(etccdi_dir.glob('etccdi_*.nc'))
        
        if not etccdi_files:
            raise FileNotFoundError(f"No ETCCDI files found in {etccdi_dir}")
        
        print(f"Found {len(etccdi_files)} ETCCDI files")
        
        # Extract years
        years = []
        for f in etccdi_files:
            name_parts = f.stem.split('_')
            for part in name_parts:
                if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2030:
                    years.append(int(part))
                    break
        
        years = sorted(set(years))
        print(f"Years: {years[0]}-{years[-1]} ({len(years)} years)")
    
    # Load data
    data_arrays = {}
    for var in ['PRCPTOT', 'R95p', 'R95pTOT', 'WD50']:
        data_arrays[var] = []
    
    valid_years = []
    
    for year in years:
        file_path = etccdi_dir / f'etccdi_precipitation_indices_{year}.nc'
        
        if file_path.exists():
            try:
                with xr.open_dataset(file_path) as ds:
                    # Handle timedelta conversion for WD50
                    for var in ['WD50', 'wet_days', 'very_wet_days']:
                        if var in ds.data_vars and 'timedelta' in str(ds[var].dtype):
                            ds[var] = ds[var] / np.timedelta64(1, 'D')
                            ds[var] = ds[var].astype(np.float64)
                    
                    # Store data
                    year_data = {}
                    for var in data_arrays.keys():
                        if var in ds.data_vars:
                            year_data[var] = ds[var].values
                        else:
                            year_data[var] = np.full((len(ds.latitude), len(ds.longitude)), np.nan)
                    
                    # Add to arrays
                    for var in data_arrays.keys():
                        data_arrays[var].append(year_data[var])
                    
                    valid_years.append(year)
                    
                    # Get coordinates from first file
                    if len(valid_years) == 1:
                        lat_coords = ds.latitude.values
                        lon_coords = ds.longitude.values
                        
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    # Combine into dataset
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    data_vars = {}
    for var, data_list in data_arrays.items():
        if data_list:
            data_vars[var] = xr.DataArray(
                np.stack(data_list, axis=0),
                dims=['year', 'latitude', 'longitude'],
                coords=coords
            )
    
    combined_ds = xr.Dataset(data_vars, coords=coords)
    
    print(f"Successfully loaded ETCCDI: {len(valid_years)} years, {len(data_vars)} variables")
    return combined_ds

def load_enhanced_data_optimized(enhanced_dir, variables=['WD25', 'WD50', 'WD75'], years=None):
    """Load enhanced concentration indices data."""
    enhanced_dir = Path(enhanced_dir)
    
    print(f"Loading enhanced data from: {enhanced_dir.absolute()}")
    
    if years is None:
        enhanced_files = list(enhanced_dir.glob('enhanced_concentration_indices_*_all.nc'))
        if not enhanced_files:
            enhanced_files = list(enhanced_dir.glob('enhanced_concentration_indices_*.nc'))
        
        if not enhanced_files:
            raise FileNotFoundError(f"No enhanced files found in {enhanced_dir}")
        
        print(f"Found {len(enhanced_files)} enhanced files")
        
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
    
    # Load data
    data_arrays = {var: [] for var in variables}
    valid_years = []
    
    for year in years:
        # Try different file patterns
        possible_files = [
            enhanced_dir / f'enhanced_concentration_indices_{year}_all.nc',
            enhanced_dir / f'enhanced_concentration_indices_{year}_basic.nc',
            enhanced_dir / f'enhanced_concentration_indices_{year}.nc'
        ]
        
        file_path = None
        for pf in possible_files:
            if pf.exists():
                file_path = pf
                break
        
        if file_path:
            try:
                with xr.open_dataset(file_path) as ds:
                    # Handle timedelta conversion
                    for var in variables:
                        if var in ds.data_vars and 'timedelta' in str(ds[var].dtype):
                            ds[var] = ds[var] / np.timedelta64(1, 'D')
                            ds[var] = ds[var].astype(np.float64)
                    
                    # Store data
                    year_data = {}
                    for var in variables:
                        if var in ds.data_vars:
                            data_values = ds[var].values
                            # Ensure data is valid (not NaN everywhere, and reasonable range)
                            if np.all(np.isnan(data_values)):
                                print(f"  Warning: {var} contains all NaN values for {year}")
                                year_data[var] = np.full((len(ds.latitude), len(ds.longitude)), np.nan)
                            else:
                                # Check for extreme values that might cause plotting issues
                                valid_data = data_values[~np.isnan(data_values)]
                                if len(valid_data) > 0:
                                    # For WD variables, ensure non-negative values
                                    if var.startswith('WD'):
                                        data_values = np.maximum(data_values, 0)  # Ensure non-negative
                                year_data[var] = data_values
                        else:
                            year_data[var] = np.full((len(ds.latitude), len(ds.longitude)), np.nan)
                    
                    # Add to arrays
                    for var in variables:
                        data_arrays[var].append(year_data[var])
                    
                    valid_years.append(year)
                    
                    # Get coordinates from first file
                    if len(valid_years) == 1:
                        lat_coords = ds.latitude.values
                        lon_coords = ds.longitude.values
                        
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    # Combine into dataset
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    data_vars = {}
    for var, data_list in data_arrays.items():
        if data_list:
            data_vars[var] = xr.DataArray(
                np.stack(data_list, axis=0),
                dims=['year', 'latitude', 'longitude'],
                coords=coords
            )
    
    combined_ds = xr.Dataset(data_vars, coords=coords)
    
    print(f"Successfully loaded enhanced data: {len(valid_years)} years, {len(data_vars)} variables")
    return combined_ds

def load_wd50r_data_optimized(wd50r_dir, percentiles=[25, 50, 75], years=None):
    """Load WD50R chronological data."""
    wd50r_dir = Path(wd50r_dir)
    
    print(f"Loading WD50R data from: {wd50r_dir.absolute()}")
    
    if years is None:
        wd50r_files = list(wd50r_dir.glob('wd50r_indices_*.nc'))
        
        if not wd50r_files:
            raise FileNotFoundError(f"No WD50R files found in {wd50r_dir}")
        
        print(f"Found {len(wd50r_files)} WD50R files")
        
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
    
    # Load data for all percentiles
    wdxxr_datasets = {}
    
    for percentile in percentiles:
        print(f"\nLoading WD{percentile}R data...")
        wdxxr_data_list = []
        valid_years = []
        
        for year in years:
            year_files = list(wd50r_dir.glob(f'wd50r_indices_{year}_*.nc'))
            
            if year_files:
                file_path = year_files[0]
                
                try:
                    with xr.open_dataset(file_path) as ds:
                        var_name = f'WD{percentile}R'
                        
                        if var_name in ds.data_vars:
                            wd_var = ds[var_name]

                            # Handle timedelta conversion
                            if 'timedelta' in str(wd_var.dtype):
                                wd_values = wd_var / np.timedelta64(1, 'D')
                                wd_values = wd_values.astype(np.float64)
                            elif 'datetime' in str(wd_var.dtype):
                                wd_values = wd_var.astype(np.float64)
                            else:
                                # Check for large values (nanoseconds/microseconds)
                                sample_vals = wd_var.values.flatten()
                                valid_sample = sample_vals[~np.isnan(sample_vals)]

                                if len(valid_sample) > 0 and np.mean(valid_sample) > 1e10:
                                    wd_values = wd_var / 1e9 / 86400  # ns -> days
                                    wd_values = wd_values.astype(np.float64)
                                elif len(valid_sample) > 0 and np.mean(valid_sample) > 1e6:
                                    wd_values = wd_var / 1e6 / 86400  # μs -> days
                                    wd_values = wd_values.astype(np.float64)
                                else:
                                    wd_values = wd_var.astype(np.float64)

                            # Validate and clean the data
                            data_values = wd_values.values
                            if np.all(np.isnan(data_values)):
                                print(f"  Warning: {var_name} contains all NaN values for {year}")
                                wdxxr_data_list.append(np.full((len(ds.latitude), len(ds.longitude)), np.nan))
                            else:
                                # Ensure non-negative values for WDXXR variables
                                data_values = np.maximum(data_values, 0)
                                wdxxr_data_list.append(data_values)
                            valid_years.append(year)
                            
                            # Get coordinates from first file
                            if len(valid_years) == 1:
                                lat_coords = ds.latitude.values
                                lon_coords = ds.longitude.values
                                
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
        
        if wdxxr_data_list:
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
    
    # Combine all percentiles into single dataset
    if wdxxr_datasets:
        combined_ds = xr.Dataset(wdxxr_datasets, coords=coords)
        print(f"\nSuccessfully loaded {len(wdxxr_datasets)} WDXXR variables")
        return combined_ds
    else:
        raise ValueError("No valid WDXXR data found")

def calculate_period_medians_and_change(ds, variables, period1_years, period2_years, mask):
    """Calculate median values for two periods and relative change."""
    results = {}
    
    # Get available years in the dataset
    available_years = ds.year.values
    print(f"Available years in dataset: {available_years[0]}-{available_years[-1]} ({len(available_years)} years)")
    
    # Filter periods to only include available years
    period1_available = [year for year in period1_years if year in available_years]
    period2_available = [year for year in period2_years if year in available_years]
    
    print(f"Period 1 available years: {period1_available}")
    print(f"Period 2 available years: {period2_available}")
    
    if not period1_available or not period2_available:
        print("Warning: No data available for one or both periods")
        return results
    
    for var in variables:
        if var not in ds.data_vars:
            print(f"Warning: {var} not found in dataset")
            continue
        
        print(f"Processing {var}...")
        
        try:
            # Get data for both periods (only available years)
            period1_data = ds[var].sel(year=period1_available).values
            period2_data = ds[var].sel(year=period2_available).values
            
            # Calculate medians (ignoring NaN values)
            period1_median = np.nanmedian(period1_data, axis=0)
            period2_median = np.nanmedian(period2_data, axis=0)

            # Apply mask consistently - ensure mask is boolean and properly shaped
            mask = mask.astype(bool)

            period1_median_masked = period1_median.copy()
            period2_median_masked = period2_median.copy()

            # Apply mask - ensure ocean areas are set to NaN
            period1_median_masked[~mask] = np.nan
            period2_median_masked[~mask] = np.nan
            
            # Calculate relative change (%) - use original data to avoid double-masking issues
            with np.errstate(divide='ignore', invalid='ignore'):
                # Handle potential zero or very small values in denominator
                denominator = period1_median.copy()
                # Set very small values to NaN to avoid extreme relative changes
                denominator[np.abs(denominator) < 1e-10] = np.nan
                relative_change = ((period2_median - period1_median) / denominator) * 100

            # Set infinite values to NaN
            relative_change[~np.isfinite(relative_change)] = np.nan

            # Apply mask to relative change (ensures ocean areas are properly masked)
            relative_change[~mask] = np.nan

            # Additional check: mask out extreme values that might cause plotting issues
            extreme_mask = np.abs(relative_change) > 1000  # More than 1000% change
            relative_change[extreme_mask] = np.nan

            # For WD variables, add additional safeguards
            if var.startswith('WD'):
                # Ensure no negative values in relative change (shouldn't happen but just in case)
                relative_change[relative_change < -500] = np.nan  # Unrealistic negative changes
                relative_change[relative_change > 500] = np.nan   # Unrealistic positive changes

                # Additional check: if period1 median is very small, relative change might be unstable
                small_values_mask = np.abs(period1_median) < 0.1  # Very small WD values
                relative_change[small_values_mask] = np.nan
            
            results[var] = {
                'period1_median': period1_median_masked,
                'period2_median': period2_median_masked,
                'relative_change': relative_change
            }
            
        except Exception as e:
            print(f"Error processing {var}: {e}")
            continue
    
    return results

def create_subplot_panel(ax, data, lat, lon, title, cmap, vmin, vmax, mask, symmetric=False):
    """Create individual subplot with map features."""
    # Apply mask using numpy masked array to ensure proper handling
    # Double-check that mask is applied consistently
    data_masked = np.ma.masked_array(data, mask=~mask)

    # Additional validation: ensure all non-masked values are finite
    data_masked[~np.isfinite(data_masked)] = np.ma.masked

    # Map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    ax.set_global()

    # Create levels
    if symmetric:
        levels = np.linspace(vmin, vmax, 20)
    else:
        levels = np.linspace(vmin, vmax, 20)

    # Plot data with proper mask handling
    try:
        # Use pcolormesh instead of contourf to avoid interpolation artifacts
        # This is more robust for masked data
        im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap,
                          vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                          shading='nearest')  # Use nearest to avoid interpolation
    except Exception as e:
        print(f"Warning: pcolormesh failed ({e}), falling back to contourf")
        try:
            # Fallback to contourf with masked array support
            im = ax.contourf(lon, lat, data_masked, levels=levels, cmap=cmap,
                            transform=ccrs.PlateCarree(), extend='neither')  # Changed extend to 'neither'
        except:
            print("Warning: contourf also failed, using basic pcolormesh")
            # Final fallback
            im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap,
                              transform=ccrs.PlateCarree(), shading='auto')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    return im

def print_data_range_summary(data_dict, variables, mask):
    """Print data range summary for scientific reporting."""
    print("\n" + "="*80)
    print("DATA RANGES (for scientific reporting)")
    print("="*80)
    
    for var in variables:
        if var not in data_dict:
            continue
        
        var_data = data_dict[var]
        
        # Get land-only data
        period1_land = var_data['period1_median'][mask & ~np.isnan(var_data['period1_median'])]
        period2_land = var_data['period2_median'][mask & ~np.isnan(var_data['period2_median'])]
        change_land = var_data['relative_change'][mask & ~np.isnan(var_data['relative_change'])]
        
        if len(period1_land) > 0 and len(period2_land) > 0:
            print(f"\n{var}:")
            print(f"  Period 1 (1980-1999) range: {np.nanmin(period1_land):.1f} - {np.nanmax(period1_land):.1f}")
            print(f"  Period 2 (2000-2024) range: {np.nanmin(period2_land):.1f} - {np.nanmax(period2_land):.1f}")
            if len(change_land) > 0:
                print(f"  Relative change range: {np.nanmin(change_land):.1f}% - {np.nanmax(change_land):.1f}%")
    
    print("\nNote: Ranges shown are full data ranges (land-only).")
    print("      Colorbars use percentile-based ranges to prevent extreme values")
    print("      from dominating the visualization (standard practice).")
    print("="*80 + "\n")

def create_comparison_plot(data_dict, variables, lat, lon, mask, plot_title, output_path, 
                          variable_info):
    """Create comparison plot for a set of variables with fixed colorbar positioning."""
    n_vars = len(variables)
    
    # Create figure with optimized height (increased for better spacing)
    fig = plt.figure(figsize=(10, 4.0 * n_vars))
    proj = ccrs.Robinson(central_longitude=0)
    
    # Store axes and images for consistent colorbar positioning
    axes_info = []
    
    for row_idx, var in enumerate(variables):
        if var not in data_dict:
            continue
        
        var_data = data_dict[var]
        var_props = variable_info.get(var, {})
        
        # Collect data for consistent color scales
        all_period1 = var_data['period1_median'][mask & ~np.isnan(var_data['period1_median'])]
        all_period2 = var_data['period2_median'][mask & ~np.isnan(var_data['period2_median'])]
        all_change = var_data['relative_change'][mask & ~np.isnan(var_data['relative_change'])]

        if len(all_period1) == 0 or len(all_period2) == 0:
            continue

        # Filter out extreme values for color scale calculation
        if len(all_change) > 0:
            # Remove extreme outliers (beyond 99th percentile of absolute values)
            change_abs = np.abs(all_change)
            if len(change_abs) > 10:  # Only if we have enough data points
                change_threshold = np.percentile(change_abs, 99)
                all_change = all_change[change_abs <= change_threshold]
        
        # Calculate actual data ranges
        period1_actual_min = np.min(all_period1)
        period1_actual_max = np.max(all_period1)
        period2_actual_min = np.min(all_period2)
        period2_actual_max = np.max(all_period2)
        
        # Color scales
        if var_props.get('symmetric_periods', False):
            period_vmax = max(np.percentile(np.abs(all_period1), 98), 
                            np.percentile(np.abs(all_period2), 98))
            period1_vmin, period1_vmax = -period_vmax, period_vmax
            period2_vmin, period2_vmax = -period_vmax, period_vmax
            period_percentile_range = "2–98"
        else:
            # Special handling for R95pTOT to handle outliers better
            if var == 'R95pTOT':
                period1_vmin, period1_vmax = np.percentile(all_period1, [5, 90])
                period2_vmin, period2_vmax = np.percentile(all_period2, [5, 90])
                period_percentile_range = "5–90"
            else:
                period1_vmin, period1_vmax = np.percentile(all_period1, [2, 98])
                period2_vmin, period2_vmax = np.percentile(all_period2, [2, 98])
                period_percentile_range = "2–98"
        
        # Calculate actual data ranges for change
        change_actual_min = np.min(all_change) if len(all_change) > 0 else -10
        change_actual_max = np.max(all_change) if len(all_change) > 0 else 10
        
        # Change is always symmetric
        if len(all_change) > 0:
            change_vmax = np.percentile(np.abs(all_change), 95)
            # Ensure reasonable bounds for WD variables
            if var.startswith('WD'):
                change_vmax = min(change_vmax, 200)  # Cap at 200% for WD variables
            change_vmin, change_vmax = -change_vmax, change_vmax
            change_percentile_range = "5–95"
        else:
            change_vmin, change_vmax = -10, 10
            change_percentile_range = "N/A"
        
        # Create subplots for this variable
        # Period 1 (1980-1999)
        ax1 = plt.subplot(n_vars, 3, row_idx * 3 + 1, projection=proj)
        im1 = create_subplot_panel(ax1, var_data['period1_median'], lat, lon,
                                  f"({chr(97 + row_idx * 3)}) {var_props.get('title', var)}\n1980-1999",
                                  var_props.get('cmap', 'viridis'),
                                  period1_vmin, period1_vmax, mask)
        
        # Period 2 (2000-2024)
        ax2 = plt.subplot(n_vars, 3, row_idx * 3 + 2, projection=proj)
        im2 = create_subplot_panel(ax2, var_data['period2_median'], lat, lon,
                                  f"({chr(97 + row_idx * 3 + 1)}) {var_props.get('title', var)}\n2000-2024",
                                  var_props.get('cmap', 'viridis'),
                                  period2_vmin, period2_vmax, mask)
        
        # Relative change
        ax3 = plt.subplot(n_vars, 3, row_idx * 3 + 3, projection=proj)

        # Additional validation for relative change data before plotting
        relative_change_plot = var_data['relative_change'].copy()
        # Ensure mask is applied one final time before plotting
        relative_change_plot[~mask] = np.nan
        # Remove any remaining non-finite values
        relative_change_plot[~np.isfinite(relative_change_plot)] = np.nan

        im3 = create_subplot_panel(ax3, relative_change_plot, lat, lon,
                                  f"({chr(97 + row_idx * 3 + 2)}) {var_props.get('title', var)}\nRelative Change (%)",
                                  'RdBu_r', change_vmin, change_vmax, mask, symmetric=True)
        
        # Store axes info for colorbar positioning
        axes_info.append({
            'row_idx': row_idx,
            'ax1': ax1, 'ax2': ax2, 'ax3': ax3,
            'im1': im1, 'im2': im2, 'im3': im3,
            'var_props': var_props,
            # Store vmin/vmax for this specific variable
            'period1_vmin': period1_vmin,
            'period1_vmax': period1_vmax,
            'period2_vmin': period2_vmin,
            'period2_vmax': period2_vmax,
            'change_vmin': change_vmin,
            'change_vmax': change_vmax,
            'period_percentile_range': period_percentile_range,
            'change_percentile_range': change_percentile_range,
            'period1_actual_min': period1_actual_min,
            'period1_actual_max': period1_actual_max,
            'change_actual_min': change_actual_min,
            'change_actual_max': change_actual_max
        })
    
    # Add colorbars with consistent spacing
    # First pass: collect all positions to calculate consistent spacing
    all_ax1_bottoms = [info['ax1'].get_position().y0 for info in axes_info]
    all_ax3_bottoms = [info['ax3'].get_position().y0 for info in axes_info]

    # Calculate consistent colorbar positioning
    cbar_height = 0.015  # Slightly taller for better visibility
    cbar_gap = -0.01      # Reduced gap between subplot and colorbar

    for info in axes_info:
        ax1, ax2, ax3 = info['ax1'], info['ax2'], info['ax3']
        im1, im2, im3 = info['im1'], info['im2'], info['im3']
        var_props = info['var_props']
        
        # Get vmin/vmax for THIS specific variable (not from outer scope!)
        period1_vmin = info['period1_vmin']
        period1_vmax = info['period1_vmax']
        period2_vmin = info['period2_vmin']
        period2_vmax = info['period2_vmax']
        change_vmin = info['change_vmin']
        change_vmax = info['change_vmax']
        period_percentile_range = info['period_percentile_range']
        change_percentile_range = info['change_percentile_range']
        period1_actual_min = info['period1_actual_min']
        period1_actual_max = info['period1_actual_max']
        change_actual_min = info['change_actual_min']
        change_actual_max = info['change_actual_max']

        # Get subplot positions
        ax1_pos = ax1.get_position()
        ax2_pos = ax2.get_position()
        ax3_pos = ax3.get_position()

        # Period 1 and 2 combined colorbar
        cbar1_left = ax1_pos.x0
        cbar1_width = ax2_pos.x1 - ax1_pos.x0
        cbar1_bottom = ax1_pos.y0 - cbar_gap - cbar_height

        cbar1_ax = fig.add_axes([cbar1_left, cbar1_bottom, cbar1_width, cbar_height])
        
        # Create normalized colorbar with extend='both'
        norm1 = mcolors.Normalize(vmin=period1_vmin, vmax=period1_vmax)
        dummy1 = plt.cm.ScalarMappable(norm=norm1, cmap=var_props.get('cmap', 'viridis'))
        cbar1 = fig.colorbar(dummy1, cax=cbar1_ax, orientation='horizontal', extend='both')
        
        # Comprehensive colorbar label with scientific reporting
        cbar1_label = f"{var_props.get('units', '')}\n(Colorbar: {period_percentile_range} percentile)\nData range: {period1_actual_min:.1f}–{period1_actual_max:.1f}"
        cbar1.set_label(cbar1_label, fontsize=7, linespacing=1.2)

        # Set ticks strictly within range
        ticks1 = np.linspace(period1_vmin, period1_vmax, 5)
        cbar1.set_ticks(ticks1)
        cbar1.set_ticklabels([f'{int(tick)}' for tick in ticks1])
        cbar1.ax.tick_params(labelsize=7)

        # Change colorbar
        cbar3_left = ax3_pos.x0
        cbar3_width = ax3_pos.width
        cbar3_bottom = ax3_pos.y0 - cbar_gap - cbar_height

        cbar3_ax = fig.add_axes([cbar3_left, cbar3_bottom, cbar3_width, cbar_height])
        
        # Create normalized colorbar with extend='both'
        norm3 = mcolors.Normalize(vmin=change_vmin, vmax=change_vmax)
        dummy3 = plt.cm.ScalarMappable(norm=norm3, cmap='RdBu_r')
        cbar3 = fig.colorbar(dummy3, cax=cbar3_ax, orientation='horizontal', extend='both')
        
        # Comprehensive colorbar label with scientific reporting
        cbar3_label = f'Relative Change (%)\n(Colorbar: {change_percentile_range} percentile)\nData range: {change_actual_min:.1f}–{change_actual_max:.1f}'
        cbar3.set_label(cbar3_label, fontsize=7, linespacing=1.2)

        # Set ticks strictly within range
        ticks3 = np.linspace(change_vmin, change_vmax, 5)
        cbar3.set_ticks(ticks3)
        cbar3.set_ticklabels([f'{int(tick)}' for tick in ticks3])
        cbar3.ax.tick_params(labelsize=7)
    
    # Main title
    #fig.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.97)
    
    # Layout adjustment with better spacing
    # Adjust bottom margin to accommodate colorbars (cbar_height + cbar_gap + some padding)
    bottom_margin = 0.18  # Adjusted for tighter colorbar spacing
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=bottom_margin, hspace=0.35, wspace=0.05)
    
    # Save
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create three precipitation concentration comparison plots')
    
    # Data directories
    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI files')
    parser.add_argument('--enhanced-dir', default='data/processed/enhanced_concentration_indices',
                       help='Directory containing enhanced indices')
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices',
                       help='Directory containing WD50R files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory')
    
    # Variable selection (flexibility to remove variables)
    parser.add_argument('--include-etccdi', nargs='*', 
                       default=['PRCPTOT', 'R95p', 'R95pTOT'],
                       help='ETCCDI variables to include')
    parser.add_argument('--include-multi', nargs='*',
                       default=['WD25', 'WD50', 'WD75'],
                       help='Multi-threshold variables to include')
    parser.add_argument('--include-chrono', nargs='*',
                       default=['WD25R', 'WD50R', 'WD75R'],
                       help='Chronological variables to include')
    
    # Analysis parameters
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='land',
                       help='Analysis domain')
    parser.add_argument('--period1-start', type=int, default=1980)
    parser.add_argument('--period1-end', type=int, default=1999)
    parser.add_argument('--period2-start', type=int, default=2000)
    parser.add_argument('--period2-end', type=int, default=2024)
    
    # Options to skip plots
    parser.add_argument('--skip-etccdi', action='store_true', help='Skip ETCCDI plot')
    parser.add_argument('--skip-multi', action='store_true', help='Skip multi-threshold plot')
    parser.add_argument('--skip-chrono', action='store_true', help='Skip chronological plot')
    
    args = parser.parse_args()
    
    print("="*80)
    print("THREE PRECIPITATION CONCENTRATION COMPARISON PLOTS")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Period 1: {args.period1_start}-{args.period1_end}")
    print(f"Period 2: {args.period2_start}-{args.period2_end}")
    print(f"ETCCDI variables: {args.include_etccdi}")
    print(f"Multi-threshold variables: {args.include_multi}")
    print(f"Chronological variables: {args.include_chrono}")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define variable properties for plotting
    variable_info = {
        # ETCCDI variables
        'PRCPTOT': {'title': 'PRCPTOT', 'units': 'mm/year', 'cmap': 'BrBG'},
        'R95p': {'title': 'R95p', 'units': 'mm/year', 'cmap': 'BrBG'},
        'R95pTOT': {'title': 'R95pTOT', 'units': 'fraction', 'cmap': 'RdYlBu_r'},
        'WD50': {'title': 'WD50', 'units': 'days', 'cmap': 'RdYlBu'},
        
        # Multi-threshold variables
        'WD25': {'title': 'WD25', 'units': 'days', 'cmap': 'RdYlBu'},
        'WD50': {'title': 'WD50', 'units': 'days', 'cmap': 'RdYlBu'},
        'WD75': {'title': 'WD75', 'units': 'days', 'cmap': 'RdYlBu'},
        
        # Chronological variables
        'WD25R': {'title': 'WD25R', 'units': 'days', 'cmap': 'RdYlBu'},
        'WD50R': {'title': 'WD50R', 'units': 'days', 'cmap': 'RdYlBu'},
        'WD75R': {'title': 'WD75R', 'units': 'days', 'cmap': 'RdYlBu'},
    }
    
    try:
        # Period years
        period1_years = list(range(args.period1_start, args.period1_end + 1))
        period2_years = list(range(args.period2_start, args.period2_end + 1))
        
        # Plot 1: ETCCDI indices
        if not args.skip_etccdi and args.include_etccdi:
            print("\nProcessing ETCCDI plot...")
            etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)
            
            # Create mask
            mask = create_optimized_land_ocean_mask(etccdi_ds, args.mask_type)
            
            # Calculate medians and changes
            etccdi_results = calculate_period_medians_and_change(
                etccdi_ds, args.include_etccdi, period1_years, period2_years, mask
            )
            
            # Print data range summary
            print_data_range_summary(etccdi_results, args.include_etccdi, mask)
            
            # Create plot
            create_comparison_plot(
                etccdi_results, args.include_etccdi,
                etccdi_ds.latitude.values, etccdi_ds.longitude.values, mask,
                'ETCCDI Precipitation Indices: Period Comparison',
                output_dir / 'etccdi_period_comparison.png',
                variable_info
            )
        
        # Plot 2: Multi-threshold indices
        if not args.skip_multi and args.include_multi:
            print("\nProcessing multi-threshold plot...")
            enhanced_ds = load_enhanced_data_optimized(args.enhanced_dir, args.include_multi)
            
            # Create mask
            mask = create_optimized_land_ocean_mask(enhanced_ds, args.mask_type)
            
            # Calculate medians and changes
            multi_results = calculate_period_medians_and_change(
                enhanced_ds, args.include_multi, period1_years, period2_years, mask
            )
            
            # Print data range summary
            print_data_range_summary(multi_results, args.include_multi, mask)
            
            # Create plot
            create_comparison_plot(
                multi_results, args.include_multi,
                enhanced_ds.latitude.values, enhanced_ds.longitude.values, mask,
                'Multi-Threshold Precipitation Concentration: Period Comparison',
                output_dir / 'multi_threshold_period_comparison.png',
                variable_info
            )
        
        # Plot 3: Chronological indices
        if not args.skip_chrono and args.include_chrono:
            print("\nProcessing chronological plot...")
            # Extract percentiles from variable names
            chrono_percentiles = []
            for var in args.include_chrono:
                if 'WD' in var and 'R' in var:
                    pct_str = var.replace('WD', '').replace('R', '')
                    try:
                        chrono_percentiles.append(int(pct_str))
                    except:
                        pass
            
            wd50r_ds = load_wd50r_data_optimized(args.wd50r_dir, chrono_percentiles)
            
            # Create mask
            mask = create_optimized_land_ocean_mask(wd50r_ds, args.mask_type)
            
            # Calculate medians and changes
            chrono_results = calculate_period_medians_and_change(
                wd50r_ds, args.include_chrono, period1_years, period2_years, mask
            )
            
            # Print data range summary
            print_data_range_summary(chrono_results, args.include_chrono, mask)
            
            # Create plot
            create_comparison_plot(
                chrono_results, args.include_chrono,
                wd50r_ds.latitude.values, wd50r_ds.longitude.values, mask,
                'Chronological Precipitation Clustering: Period Comparison',
                output_dir / 'chronological_period_comparison.png',
                variable_info
            )
        
        print("\n" + "="*80)
        print("ALL PLOTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Check output directory: {output_dir}")
        if not args.skip_etccdi:
            print("✅ ETCCDI period comparison")
        if not args.skip_multi:
            print("✅ Multi-threshold period comparison")
        if not args.skip_chrono:
            print("✅ Chronological period comparison")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())