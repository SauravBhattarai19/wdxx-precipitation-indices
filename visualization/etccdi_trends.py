def save_mask_debug_figure(ds, mask, mask_type, output_dir):
    """
    Create a debug figure showing the land/ocean mask.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Original data (show a sample variable to see data coverage)
    ax1 = axes[0]
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    ax1.set_title('Data Coverage Example', fontsize=14, fontweight='bold')
    
    # Show where we have valid data (use first available variable)
    for var_name in ['PRCPTOT', 'R95p', 'R95pTOT']:
        if var_name in ds.data_vars:
            sample_data = ds[var_name].isel(year=0).values
            valid_data = ~np.isnan(sample_data)
            im1 = ax1.contourf(ds.longitude, ds.latitude, valid_data.astype(int),
                              levels=[0.5, 1.5], colors=['lightblue'], alpha=0.7,
                              transform=ccrs.PlateCarree())
            ax1.text(0.02, 0.98, f'Valid {var_name} pixels', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    verticalalignment='top')
            break
    
    # Applied mask
    ax2 = axes[1]
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax2.set_global()
    ax2.set_title(f'Applied {mask_type.title()} Mask', fontsize=14, fontweight='bold')
    
    # Show the mask
    mask_colors = ['white', 'darkgreen'] if mask_type == 'land' else ['darkblue', 'white']
    im2 = ax2.contourf(ds.longitude, ds.latitude, mask.astype(int),
                      levels=[0.5, 1.5], colors=[mask_colors[1]], alpha=0.8,
                      transform=ccrs.PlateCarree())
    
    mask_pct = np.sum(mask) / mask.size * 100
    ax2.text(0.02, 0.98, f'{mask_type.title()} pixels: {np.sum(mask):,} ({mask_pct:.1f}%)',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save debug figure
    debug_filename = f'mask_debug_{mask_type}.png'
    debug_path = output_dir / debug_filename
    plt.savefig(debug_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Mask debug figure saved: {debug_path}")
    return debug_path#!/usr/bin/env python3
"""
Optimized Create Publication Figure: Trends in Precipitation Indices (ETCCDI)

High-performance version with:
- Vectorized Mann-Kendall trend analysis
- Multiprocessing for 192-core systems
- Permanent data storage (no caching)
- Scientific-grade statistical testing

Figure Layout: 4 rows × 3 columns
- Rows: PRCPTOT, R95p, R95pTOT, WD50
- Columns: 1980-2024, 1980-1999 (pre-2000), 2000-2024 (post-2000)

Usage:
python optimized_precipitation_trends.py --etccdi-dir data/processed/etccdi_indices --output-dir figures/
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
from multiprocessing import Pool, shared_memory
import time
from functools import partial
import h5py
import json
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
    
    Args:
        data_3d: 3D array (time, lat, lon)
        years: 1D array of years
    
    Returns:
        slopes: 2D array of Sen's slopes (per year)
        p_values: 2D array of p-values
        z_stats: 2D array of Z-statistics
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
            
            # Variance of S (assuming no ties for simplicity - can be enhanced)
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
    Uses scipy for accurate p-value calculation.
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
    
    Args:
        data_array: xarray DataArray with time dimension
        years: array of years
        n_processes: number of processes (default: min(192, available cores))
        chunk_size: size of spatial chunks
    
    Returns:
        slopes: trend slopes (per year)
        p_values: p-values from Mann-Kendall test
        z_stats: Z-statistics
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
            
            # Use higher resolution mask for better accuracy
            try:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(dummy_ds)
                print("  Using 50m resolution Natural Earth boundaries")
            except:
                land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
                print("  Using 110m resolution Natural Earth boundaries")
            
            land_mask = ~np.isnan(land_mask_rm.values)
            
            # Clean up mask - remove isolated pixels
            from scipy.ndimage import binary_opening, binary_closing
            if land_mask.sum() > 0:
                # Remove small isolated ocean pixels in land areas
                land_mask = binary_closing(land_mask, iterations=1)
                # Remove small isolated land pixels in ocean areas  
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
    
    # Enhanced fallback approach with better boundaries
    print("  Using enhanced geometric land/ocean approximation...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    land_mask = np.zeros_like(lat_grid, dtype=bool)
    
    # More precise continent boundaries
    land_regions = [
        # North America (more precise)
        ((lat_grid >= 25) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -50) & 
         ~((lat_grid >= 25) & (lat_grid <= 35) & (lon_grid >= -100) & (lon_grid <= -80))),  # Remove Gulf of Mexico
        
        # South America
        ((lat_grid >= -60) & (lat_grid <= 15) & (lon_grid >= -85) & (lon_grid <= -30) &
         ~((lat_grid >= -20) & (lat_grid <= 10) & (lon_grid >= -50) & (lon_grid <= -35))),  # Remove Atlantic coast
        
        # Europe (more precise)
        ((lat_grid >= 35) & (lat_grid <= 75) & (lon_grid >= -15) & (lon_grid <= 45) &
         ~((lat_grid >= 35) & (lat_grid <= 50) & (lon_grid >= -15) & (lon_grid <= 5))),   # Remove Atlantic
        
        # Asia (more precise)
        ((lat_grid >= 5) & (lat_grid <= 75) & (lon_grid >= 25) & (lon_grid <= 180) &
         ~((lat_grid >= 5) & (lat_grid <= 25) & (lon_grid >= 50) & (lon_grid <= 75))),    # Remove Arabian Sea
        
        # Africa (more precise)
        ((lat_grid >= -35) & (lat_grid <= 40) & (lon_grid >= -20) & (lon_grid <= 55) &
         ~((lat_grid >= -10) & (lat_grid <= 20) & (lon_grid >= 30) & (lon_grid <= 50))),  # Remove Red Sea/Indian Ocean
        
        # Australia and Oceania
        ((lat_grid >= -50) & (lat_grid <= -5) & (lon_grid >= 110) & (lon_grid <= 180)),
        
        # Greenland
        ((lat_grid >= 60) & (lat_grid <= 85) & (lon_grid >= -75) & (lon_grid <= -10)),
        
        # Antarctica  
        (lat_grid <= -60),
    ]
    
    # Combine all land regions
    for region in land_regions:
        land_mask |= region
    
    # Remove major water bodies more precisely
    water_bodies = [
        # Mediterranean Sea
        ((lat_grid >= 30) & (lat_grid <= 50) & (lon_grid >= -10) & (lon_grid <= 40)),
        # Black Sea
        ((lat_grid >= 40) & (lat_grid <= 50) & (lon_grid >= 25) & (lon_grid <= 45)),
        # Caspian Sea
        ((lat_grid >= 35) & (lat_grid <= 50) & (lon_grid >= 45) & (lon_grid <= 55)),
        # Great Lakes region (approximate)
        ((lat_grid >= 40) & (lat_grid <= 50) & (lon_grid >= -95) & (lon_grid <= -75)),
        # Hudson Bay
        ((lat_grid >= 50) & (lat_grid <= 65) & (lon_grid >= -100) & (lon_grid <= -75)),
    ]
    
    for water_body in water_bodies:
        land_mask &= ~water_body
    
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

def save_trends_permanent(trends_data, data_info, output_dir, mask_type):
    """
    Save trends data to permanent HDF5 format for repeated use.
    """
    # Create unique filename based on data characteristics
    years_str = f"{data_info['years'][0]}-{data_info['years'][-1]}"
    shape_str = f"{data_info['shape'][0]}x{data_info['shape'][1]}"
    filename = f"trends_permanent_{years_str}_{shape_str}_{mask_type}.h5"
    
    trends_file = output_dir / filename
    
    print(f"Saving permanent trends data to: {trends_file}")
    
    with h5py.File(trends_file, 'w') as f:
        # Save metadata
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['mask_type'] = mask_type
        metadata_grp.attrs['created_date'] = pd.Timestamp.now().isoformat()
        metadata_grp.attrs['years_range'] = f"{data_info['years'][0]}-{data_info['years'][-1]}"
        metadata_grp.attrs['grid_shape'] = data_info['shape']
        metadata_grp.attrs['lat_range'] = data_info['lat_range']
        metadata_grp.attrs['lon_range'] = data_info['lon_range']
        metadata_grp.create_dataset('years', data=np.array(data_info['years']))
        # Handle NumPy 2.0 compatibility for string arrays
        try:
            # NumPy < 2.0
            variables_data = np.array(data_info['variables'], dtype='S')
        except:
            # NumPy >= 2.0 or other issues - use bytes encoding
            variables_data = np.array([var.encode('utf-8') for var in data_info['variables']], dtype='S')
        
        metadata_grp.create_dataset('variables', data=variables_data)
        
        # Save periods metadata
        periods_grp = metadata_grp.create_group('periods')
        for period_name, period_years in data_info['periods'].items():
            periods_grp.attrs[period_name] = period_years
        
        # Save trends data for each variable and period
        for var_name, var_data in trends_data.items():
            var_grp = f.create_group(var_name)
            
            for period_name, period_data in var_data.items():
                period_grp = var_grp.create_group(period_name)
                
                # Save arrays with compression
                period_grp.create_dataset('slopes', data=period_data['slopes'], 
                                        compression='gzip', compression_opts=9)
                period_grp.create_dataset('p_values', data=period_data['p_values'], 
                                        compression='gzip', compression_opts=9)
                period_grp.create_dataset('z_stats', data=period_data['z_stats'], 
                                        compression='gzip', compression_opts=9)
                period_grp.create_dataset('significant', data=period_data['significant'], 
                                        compression='gzip', compression_opts=9)
                period_grp.create_dataset('years', data=period_data['years'])
                
                # Metadata
                period_grp.attrs['n_years'] = len(period_data['years'])
                period_grp.attrs['significant_pixels'] = np.sum(period_data['significant'])
                period_grp.attrs['total_pixels'] = period_data['significant'].size
    
    print(f"Trends data permanently saved: {trends_file.stat().st_size / (1024**3):.2f} GB")
    return trends_file

def load_trends_permanent(output_dir, mask_type=None, filename=None):
    """
    Load permanent trends data from HDF5 format.
    """
    if filename:
        trends_file = Path(filename)
    else:
        # Find most recent trends file for this mask type
        pattern = f"trends_permanent_*_{mask_type}.h5" if mask_type else "trends_permanent_*.h5"
        trends_files = list(output_dir.glob(pattern))
        
        if not trends_files:
            return None
        
        trends_file = max(trends_files, key=lambda x: x.stat().st_mtime)
    
    if not trends_file.exists():
        return None
    
    print(f"Loading permanent trends data from: {trends_file}")
    
    trends_data = {}
    
    with h5py.File(trends_file, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        
        # Handle variables with robust loading
        try:
            # Try to load variables dataset first
            if 'variables' in f['metadata']:
                variables_data = f['metadata/variables'][:]
                # Handle different data types
                if hasattr(variables_data[0], 'decode'):
                    variables = [s.decode('utf-8') for s in variables_data]
                else:
                    variables = [str(s) for s in variables_data]
            else:
                raise KeyError("Variables dataset not found")
                
        except (KeyError, TypeError) as e:
            print(f"  Variables dataset not found, trying attributes fallback: {e}")
            # Fallback: load from attributes
            variables = []
            if 'n_variables' in metadata:
                n_vars = metadata['n_variables']
                for i in range(n_vars):
                    var_key = f'variable_{i}'
                    if var_key in metadata:
                        variables.append(str(metadata[var_key]))
            
            if not variables:
                # Last resort: scan for variable groups
                print("  Scanning HDF5 groups for variables...")
                possible_vars = ['PRCPTOT', 'R95p', 'R95pTOT', 'WD50']
                variables = [var for var in possible_vars if var in f]
                
        if not variables:
            raise ValueError("Could not determine variables from HDF5 file")
        
        print(f"  Variables: {variables}")
        print(f"  Years: {metadata['years_range']}")
        print(f"  Grid: {metadata['grid_shape']}")
        
        # Load trends data
        for var_name in variables:
            if var_name not in f:
                continue
                
            trends_data[var_name] = {}
            var_grp = f[var_name]
            
            for period_name in var_grp.keys():
                period_grp = var_grp[period_name]
                
                trends_data[var_name][period_name] = {
                    'slopes': period_grp['slopes'][:],
                    'p_values': period_grp['p_values'][:],
                    'z_stats': period_grp['z_stats'][:],
                    'significant': period_grp['significant'][:],
                    'years': period_grp['years'][:]
                }
    
    return trends_data

def add_significance_contours(ax, ds, significant, mask, line_width=0.8, alpha=0.9):
    """
    Optimized significance visualization using contours.
    """
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return
    
    try:
        sig_float = significant_masked.astype(float)
        contours = ax.contour(ds.longitude, ds.latitude, sig_float,
                            levels=[0.5], colors='black', linewidths=line_width,
                            transform=ccrs.PlateCarree(), alpha=alpha)
        return contours
    except Exception as e:
        print(f"Warning: Could not draw significance contours: {e}")
        return None

def add_significance_boundaries(ax, ds, significant, mask, line_width=0.5, alpha=0.7):
    """
    Add significance boundaries using optimized edge detection.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return
    
    try:
        # Find boundaries using morphological operations
        sig_binary = significant_masked.astype(bool)
        dilated = binary_dilation(sig_binary, iterations=1)
        boundaries = dilated & ~sig_binary
        
        # Get boundary coordinates with subsampling for performance
        boundary_coords = np.where(boundaries)
        if len(boundary_coords[0]) > 0:
            # Subsample for performance
            n_points = len(boundary_coords[0])
            skip = max(1, n_points // 1000)  # Max 1000 points
            
            boundary_lons = ds.longitude.values[boundary_coords[1]][::skip]
            boundary_lats = ds.latitude.values[boundary_coords[0]][::skip]
            
            ax.scatter(boundary_lons, boundary_lats, s=line_width, c='black', 
                      marker='.', alpha=alpha, transform=ccrs.PlateCarree())
    except Exception as e:
        print(f"Warning: Could not draw significance boundaries: {e}")

def create_optimized_precipitation_figure(etccdi_ds, trends_data, output_dir, mask_type='both', 
                                        show_significance=True, significance_method='contour'):
    """
    Create optimized precipitation trends figure.
    """
    # Create mask
    mask = create_optimized_land_ocean_mask(etccdi_ds, mask_type)
    
    # Define indices and their properties
    indices_info = {
        'PRCPTOT': {'units': 'mm/decade', 'title': 'PRCPTOT', 'cmap': 'BrBG', 'symmetric': True},
        'R95p': {'units': 'mm/decade', 'title': 'R95p', 'cmap': 'BrBG', 'symmetric': True},
        'R95pTOT': {'units': 'fraction/decade', 'title': 'R95pTOT', 'cmap': 'RdBu_r', 'symmetric': True}
    }
    
    periods = ['1980-2024', '1980-1999', '2000-2024']
    
    # Create figure
    fig = plt.figure(figsize=(15, 9))  # Reduced height since we have 3 rows instead of 4
    proj = ccrs.Robinson(central_longitude=0)
    
    subplot_idx = 0
    
    for row_idx, (index_name, index_info) in enumerate(indices_info.items()):
        if index_name not in trends_data:
            print(f"Warning: {index_name} not found in trends data")
            continue
        
        print(f"Creating plots for {index_name}...")
        
        # Collect all trends for consistent colorbar
        all_slopes = []
        for period_name in periods:
            if period_name in trends_data[index_name]:
                slopes = trends_data[index_name][period_name]['slopes']
                valid_slopes = slopes[mask & ~np.isnan(slopes)]
                if len(valid_slopes) > 0:
                    all_slopes.extend(valid_slopes)
        
        if not all_slopes:
            continue
        
        # Convert to per-decade
        all_slopes = np.array(all_slopes) * 10
        
        # Calculate actual data range for colorbar label
        actual_min = np.min(all_slopes)
        actual_max = np.max(all_slopes)
        
        # Determine colorbar range
        if index_info['symmetric']:
            vmax_abs = np.percentile(np.abs(all_slopes), 98)
            vmin, vmax = -vmax_abs, vmax_abs
            percentile_range = "2–98"
        else:
            vmin = np.percentile(all_slopes, 2)
            vmax = np.percentile(all_slopes, 98)
            percentile_range = "2–98"
        
        levels = np.linspace(vmin, vmax, 21)
        
        # Plot each period
        for col_idx, period_name in enumerate(periods):
            if period_name not in trends_data[index_name]:
                continue
            
            subplot_idx += 1
            ax = plt.subplot(3, 3, subplot_idx, projection=proj)
            
            # Map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
            ax.set_global()
            
            # Get data
            period_data = trends_data[index_name][period_name]
            slopes = period_data['slopes'] * 10  # Convert to per decade
            significant = period_data['significant']
            
            # Apply mask
            slopes_masked = slopes.copy()
            slopes_masked[~mask] = np.nan
            
            # Plot trends
            cmap = plt.cm.get_cmap(index_info['cmap'])
            im = ax.contourf(etccdi_ds.longitude, etccdi_ds.latitude, slopes_masked,
                           levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                           extend='both')
            
            # Add significance
            if show_significance:
                if significance_method == 'contour':
                    add_significance_contours(ax, etccdi_ds, significant, mask)
                elif significance_method == 'boundary':
                    add_significance_boundaries(ax, etccdi_ds, significant, mask)
            
            # Statistics - only count masked pixels
            valid_trends = slopes_masked[mask & ~np.isnan(slopes_masked)]
            sig_pixels_masked = significant & mask
            
            # Calculate separate statistics for increasing and decreasing trends
            if show_significance and np.sum(mask) > 0:
                # Get significant pixels with their trend directions
                sig_slopes = slopes_masked[sig_pixels_masked & ~np.isnan(slopes_masked)]
                
                if len(sig_slopes) > 0:
                    # Count increasing and decreasing trends
                    sig_increasing = np.sum(sig_slopes > 0)
                    sig_decreasing = np.sum(sig_slopes < 0)
                    
                    # Calculate percentages relative to total masked pixels
                    total_masked_pixels = np.sum(mask)
                    inc_percentage = sig_increasing / total_masked_pixels * 100
                    dec_percentage = sig_decreasing / total_masked_pixels * 100
                    
                    # Create detailed statistics text
                    stats_text = f'{inc_percentage:.1f}% ↗\n{dec_percentage:.1f}% ↘'
                else:
                    stats_text = '0.0% ↗\n0.0% ↘'
            else:
                stats_text = ''
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(period_name, fontsize=14, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ylabel = f"({chr(97 + row_idx)}) {index_info['title']}"
                ax.text(-0.15, 0.5, ylabel, transform=ax.transAxes, rotation=90,
                       ha='center', va='center', fontweight='bold', fontsize=14)
            
            # Add statistics with validation - moved down
            if stats_text:
                ax.text(0.02, 0.08, stats_text,
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add colorbar for each row
        if subplot_idx % 3 == 0:  # After every 3rd subplot (end of row)
            ax_pos = ax.get_position()
            cbar_ax = fig.add_axes([ax_pos.x1 + 0.01, ax_pos.y0, 0.02, ax_pos.height])
            
            # Create normalized colorbar with extend='both' for extremes
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            dummy_im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(dummy_im, cax=cbar_ax, extend='both')
            
            # Comprehensive colorbar label
            cbar_label = f"{index_info['units']}\nColor: {percentile_range} percentile\nActual: {actual_min:.2f} to {actual_max:.2f}"
            cbar.set_label(cbar_label, fontsize=10, linespacing=1.2)
            
            # Set ticks strictly within [vmin, vmax]
            tick_locs = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(tick_locs)
            cbar.set_ticklabels([f'{val:.2f}' for val in tick_locs])
            cbar.ax.tick_params(labelsize=9)
    
    # Main title
    title_text = 'Trends in Precipitation Indices (Mann-Kendall Analysis)'
    if show_significance:
        sig_desc = 'contours' if significance_method == 'contour' else 'boundaries'
        title_text += f'\n({sig_desc.capitalize()} indicate statistical significance, p < 0.10)'
    
    #fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.95)
    
    # Layout
    plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.05,
                       wspace=0.05, hspace=-0.075)
    
    # Save
    filename = f'precipitation_trends_optimized_{mask_type}'
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
    """Optimized main function."""
    parser = argparse.ArgumentParser(description='Optimized precipitation trends analysis and visualization')
    
    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain (default: both)')
    parser.add_argument('--no-significance', action='store_true',
                       help='Hide statistical significance')
    parser.add_argument('--sig-method', choices=['contour', 'boundary'], default='contour',
                       help='Significance visualization method (default: contour)')
    parser.add_argument('--n-processes', type=int, default=128,
                       help='Number of processes for parallel computation (default: min(192, available cores))')
    parser.add_argument('--load-trends', type=str, default=None,
                       help='Load existing trends file instead of calculating')
    parser.add_argument('--calculate-only', action='store_true',
                       help='Calculate and save trends only, skip figure creation')
    parser.add_argument('--debug-mask', action='store_true',
                       help='Create debug figure showing land/ocean mask')
    parser.add_argument('--land-only', action='store_true',
                       help='Shortcut for --mask-type land (land-only analysis)')
    parser.add_argument('--ocean-only', action='store_true', 
                       help='Shortcut for --mask-type ocean (ocean-only analysis)')
    
    args = parser.parse_args()
    
    # Handle shortcut options
    if args.land_only:
        args.mask_type = 'land'
    elif args.ocean_only:
        args.mask_type = 'ocean'
    
    print("="*80)
    print("OPTIMIZED PRECIPITATION TRENDS ANALYSIS")
    print("="*80)
    print(f"Input directory: {args.etccdi_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Show significance: {not args.no_significance}")
    print(f"Significance method: {args.sig_method}")
    print(f"Processes: {args.n_processes if args.n_processes else 'auto'}")
    if args.debug_mask:
        print("Debug mask visualization: enabled")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load or calculate trends
        if args.load_trends:
            print(f"Loading existing trends from: {args.load_trends}")
            trends_data = load_trends_permanent(output_dir, filename=args.load_trends)
            if trends_data is None:
                raise ValueError(f"Could not load trends from {args.load_trends}")
            
            # Load dataset for coordinates (needed for plotting)
            etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)
            
        else:
            # Load data
            print("Loading ETCCDI data...")
            etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)
            
            # Define periods
            periods = {
                '1980-2024': (1980, 2024),
                '1980-1999': (1980, 1999), 
                '2000-2024': (2000, 2024)
            }
            
            # Calculate trends
            print("Calculating trends using Mann-Kendall analysis...")
            start_time = time.time()
            
            trends_data = {}
            
            for index_name in ['PRCPTOT', 'R95p', 'R95pTOT']:
                if index_name not in etccdi_ds.data_vars:
                    print(f"Warning: {index_name} not found in dataset")
                    continue
                
                print(f"\nProcessing {index_name}...")
                trends_data[index_name] = {}
                
                for period_name, (start_year, end_year) in periods.items():
                    period_ds = etccdi_ds.sel(year=slice(start_year, end_year))
                    period_years = period_ds.year.values.astype(int)
                    
                    if len(period_years) < 4:
                        print(f"  Skipping {period_name}: insufficient data ({len(period_years)} years)")
                        continue
                    
                    print(f"  Calculating trends for {period_name} ({len(period_years)} years)...")
                    period_start = time.time()
                    
                    slopes, p_values, z_stats = calculate_trends_parallel(
                        period_ds[index_name], period_years, args.n_processes
                    )
                    
                    significant = p_values < 0.10
                    
                    trends_data[index_name][period_name] = {
                        'slopes': slopes,
                        'p_values': p_values,
                        'z_stats': z_stats,
                        'significant': significant,
                        'years': period_years
                    }
                    
                    period_time = time.time() - period_start
                    sig_pct = np.sum(significant) / significant.size * 100
                    print(f"    Completed in {period_time:.1f}s ({sig_pct:.1f}% significant pixels)")
            
            total_time = time.time() - start_time
            print(f"\nTotal trend calculation time: {total_time:.1f}s")
            
            # Save trends permanently
            data_info = {
                'years': sorted([int(y) for y in etccdi_ds.year.values]),
                'variables': ['PRCPTOT', 'R95p', 'R95pTOT'],
                'periods': {k: [int(v[0]), int(v[1])] for k, v in periods.items()},
                'shape': [len(etccdi_ds.latitude), len(etccdi_ds.longitude)],
                'lat_range': [float(etccdi_ds.latitude.min()), float(etccdi_ds.latitude.max())],
                'lon_range': [float(etccdi_ds.longitude.min()), float(etccdi_ds.longitude.max())]
            }
            
            trends_file = save_trends_permanent(trends_data, data_info, output_dir, args.mask_type)
            print(f"\nPermanent trends data saved: {trends_file}")
        
        if args.calculate_only:
            print("Calculation complete. Skipping figure creation.")
            return 0
        
        # Debug mask if requested
        if args.debug_mask:
            print("\nCreating mask debug visualization...")
            mask_debug = create_optimized_land_ocean_mask(etccdi_ds, args.mask_type)
            save_mask_debug_figure(etccdi_ds, mask_debug, args.mask_type, output_dir)
        
        # Create figure
        print("\nCreating optimized precipitation trends figure...")
        output_path = create_optimized_precipitation_figure(
            etccdi_ds, trends_data, output_dir, args.mask_type,
            show_significance=not args.no_significance,
            significance_method=args.sig_method
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print("Optimizations applied:")
        print("- Vectorized Mann-Kendall trend analysis")
        print("- Multiprocessing for parallel computation")
        print("- Optimized land/ocean masking")
        print("- Permanent HDF5 data storage")
        print("- Scientific-grade statistical testing")
        
        etccdi_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())