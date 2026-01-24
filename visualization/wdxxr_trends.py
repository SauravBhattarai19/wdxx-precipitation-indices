#!/usr/bin/env python3
"""
Create Publication Figure: Trends in WDXXR Chronological Precipitation Indices

High-performance version with:
- Vectorized Mann-Kendall trend analysis for multiple percentiles
- Multiprocessing for 192-core systems
- Permanent data storage (no caching)
- Scientific-grade statistical testing

Figure Layout: 4 rows × 3 columns
- Rows: WD25R, WD50R, WD75R, WD90R (chronological precipitation clustering)
- Columns: 1980-2024, 1980-1999 (pre-2000), 2000-2024 (post-2000)

Usage:
python figure4.py --wd50r-dir data/processed/wd50r_indices --output-dir figures/
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
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
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
    (Same implementation as figure2.py)
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
    (Same implementation as figure2.py)
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
    (Same implementation as figure2.py)
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
    (Same implementation as figure2.py)
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
                print(f"  Land pixels: {np.sum(land_mask):,} ({land_pct:.1f}%)")
            
            return land_mask if mask_type == 'land' else ~land_mask
            
        except Exception as e:
            print(f"  regionmask failed: {e}, using fallback")
    
    # Enhanced fallback approach (same as figure2.py)
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

def load_wdxxr_data_optimized(wd50r_dir, percentiles=[25, 50, 75, 90], years=None):
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
                            
                            # Debug info for first file
                            if len(valid_years) == 0:
                                print(f"  WD{percentile}R dtype: {wd_var.dtype}")
                                sample_vals = wd_var.values.flatten()
                                valid_sample = sample_vals[~np.isnan(sample_vals)][:5]
                                print(f"  Sample values: {valid_sample}")
                            
                            # Handle timedelta conversion (same logic as figure3.py)
                            if 'timedelta' in str(wd_var.dtype):
                                #print(f"  Converting WD{percentile}R from timedelta to days")
                                wd_values = wd_var / np.timedelta64(1, 'D')
                                wd_values = wd_values.astype(np.float64)
                            elif 'datetime' in str(wd_var.dtype):
                                #print(f"  Converting WD{percentile}R from datetime to days")
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
                            
                            # Debug converted values for first file
                            if len(valid_years) == 0:
                                conv_sample = wd_values.values.flatten()
                                valid_conv = conv_sample[~np.isnan(conv_sample)][:5]
                                print(f"  Converted sample values: {valid_conv}")
                            
                            wdxxr_data_list.append(wd_values.values)
                            valid_years.append(year)
                            
                            # Get coordinates from first file
                            if len(valid_years) == 1:
                                lat_coords = ds.latitude.values
                                lon_coords = ds.longitude.values
                                
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

def save_wdxxr_trends_permanent(trends_data, data_info, output_dir, mask_type):
    """
    Save WDXXR trends data to permanent HDF5 format.
    """
    # Create unique filename
    years_str = f"{data_info['years'][0]}-{data_info['years'][-1]}"
    shape_str = f"{data_info['shape'][0]}x{data_info['shape'][1]}"
    percentiles_str = '_'.join(map(str, data_info['percentiles']))
    filename = f"wdxxr_trends_{years_str}_{shape_str}_P{percentiles_str}_{mask_type}.h5"
    
    trends_file = output_dir / filename
    
    print(f"Saving permanent WDXXR trends data to: {trends_file}")
    
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
        
        # Save percentiles
        percentiles_data = np.array([str(p) for p in data_info['percentiles']], dtype='S')
        metadata_grp.create_dataset('percentiles', data=percentiles_data)
        
        # Save periods metadata
        periods_grp = metadata_grp.create_group('periods')
        for period_name, period_years in data_info['periods'].items():
            periods_grp.attrs[period_name] = period_years
        
        # Save trends data for each percentile and period
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
    
    print(f"WDXXR trends data permanently saved: {trends_file.stat().st_size / (1024**3):.2f} GB")
    return trends_file

def load_wdxxr_trends_permanent(output_dir, mask_type=None, filename=None):
    """
    Load permanent WDXXR trends data from HDF5 format.
    """
    if filename:
        trends_file = Path(filename)
    else:
        pattern = f"wdxxr_trends_*_{mask_type}.h5" if mask_type else "wdxxr_trends_*.h5"
        trends_files = list(output_dir.glob(pattern))
        
        if not trends_files:
            return None
        
        trends_file = max(trends_files, key=lambda x: x.stat().st_mtime)
    
    if not trends_file.exists():
        return None
    
    print(f"Loading permanent WDXXR trends data from: {trends_file}")
    
    trends_data = {}
    
    with h5py.File(trends_file, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        
        # Load percentiles
        try:
            percentiles_data = f['metadata/percentiles'][:]
            percentiles = [int(s.decode('utf-8')) for s in percentiles_data]
        except:
            # Fallback: scan for percentile variables
            possible_percentiles = [25, 50, 75, 90]
            percentiles = [p for p in possible_percentiles if f'WD{p}R' in f]
        
        print(f"  Percentiles: {percentiles}")
        print(f"  Years: {metadata['years_range']}")
        print(f"  Grid: {metadata['grid_shape']}")
        
        # Load trends data
        for percentile in percentiles:
            var_name = f'WD{percentile}R'
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
    Add significance contours with proper masking.
    """
    significant_masked = significant & mask
    if not np.any(significant_masked):
        return
    
    try:
        # Apply mask more aggressively - set non-mask areas to 0
        sig_for_contour = np.zeros_like(significant_masked, dtype=float)
        sig_for_contour[significant_masked] = 1.0
        sig_for_contour[~mask] = 0.0  # Explicitly set non-mask to 0
        
        contours = ax.contour(ds.longitude, ds.latitude, sig_for_contour,
                            levels=[0.5], colors='black', linewidths=line_width,
                            transform=ccrs.PlateCarree(), alpha=alpha)
        return contours
    except Exception as e:
        print(f"Warning: Could not draw significance contours: {e}")
        return None

def save_mask_debug_figure(ds, mask, mask_type, output_dir):
    """
    Create a debug figure showing the land/ocean mask application.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Original data coverage
    ax1 = axes[0]
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax1.set_global()
    ax1.set_title('Data Coverage Example', fontsize=14, fontweight='bold')
    
    # Show where we have valid data (use first available variable)
    available_vars = [var for var in ds.data_vars if 'WD' in var and 'R' in var]
    if available_vars:
        sample_data = ds[available_vars[0]].isel(year=0).values
        valid_data = ~np.isnan(sample_data)
        im1 = ax1.contourf(ds.longitude, ds.latitude, valid_data.astype(int),
                          levels=[0.5, 1.5], colors=['lightblue'], alpha=0.7,
                          transform=ccrs.PlateCarree())
        '''
        ax1.text(0.02, 0.98, f'Valid {available_vars[0]} pixels', transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                verticalalignment='top')
                '''
    
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
    '''
    ax2.text(0.02, 0.98, f'{mask_type.title()} pixels: {np.sum(mask):,} ({mask_pct:.1f}%)',
            transform=ax2.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            verticalalignment='top')
            '''
    
    plt.tight_layout()
    
    # Save debug figure
    debug_filename = f'wdxxr_mask_debug_{mask_type}.png'
    debug_path = output_dir / debug_filename
    plt.savefig(debug_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Mask debug figure saved: {debug_path}")
    return debug_path

def create_wdxxr_trends_figure(wdxxr_ds, trends_data, output_dir, mask_type='both',
                              show_significance=True, significance_method='contour'):
    """
    Create WDXXR trends figure (4 rows × 3 columns) with proper masking.
    """
    # Create mask
    mask = create_optimized_land_ocean_mask(wdxxr_ds, mask_type)
    
    # Validate mask
    mask_pct = np.sum(mask) / mask.size * 100
    print(f"Mask validation: {np.sum(mask):,} pixels ({mask_pct:.1f}%) for {mask_type} analysis")
    
    # Define indices and their properties
    indices_info = {
        'WD25R': {'units': 'days/decade', 'title': 'WD25R', 'cmap': 'RdYlBu', 'symmetric': True},
        'WD50R': {'units': 'days/decade', 'title': 'WD50R', 'cmap': 'RdYlBu', 'symmetric': True},
        'WD75R': {'units': 'days/decade', 'title': 'WD75R', 'cmap': 'RdYlBu', 'symmetric': True},
        'WD90R': {'units': 'days/decade', 'title': 'WD90R', 'cmap': 'RdYlBu', 'symmetric': True}
    }
    
    periods = ['1980-2024', '1980-1999', '2000-2024']
    
    # Create figure
    fig = plt.figure(figsize=(15, 9))  # Taller for 4 rows
    proj = ccrs.Robinson(central_longitude=0)
    
    subplot_idx = 0
    
    for row_idx, (index_name, index_info) in enumerate(indices_info.items()):
        if index_name not in trends_data:
            print(f"Warning: {index_name} not found in trends data")
            continue
        
        print(f"Creating plots for {index_name}...")
        
        # Collect all trends for consistent colorbar - ONLY from masked pixels
        all_slopes = []
        for period_name in periods:
            if period_name in trends_data[index_name]:
                slopes = trends_data[index_name][period_name]['slopes']
                # Apply mask BEFORE collecting statistics
                slopes_temp = slopes.copy()
                slopes_temp[~mask] = np.nan
                valid_slopes = slopes_temp[mask & ~np.isnan(slopes_temp)]
                if len(valid_slopes) > 0:
                    all_slopes.extend(valid_slopes)
        
        if not all_slopes:
            print(f"  Warning: No valid {index_name} trends found within {mask_type} mask")
            continue
        
        # Convert to per-decade
        all_slopes = np.array(all_slopes) * 10
        
        # Calculate actual data range for colorbar label
        actual_min = np.min(all_slopes)
        actual_max = np.max(all_slopes)
        
        # Determine colorbar range (symmetric for trends)
        vmax_abs = np.percentile(np.abs(all_slopes), 98)
        vmin, vmax = -vmax_abs, vmax_abs
        percentile_range = "2–98"
        
        # Ensure non-zero range
        if vmax_abs < 1e-10:
            vmin, vmax = -1, 1
        
        levels = np.linspace(vmin, vmax, 21)
        
        # Plot each period
        for col_idx, period_name in enumerate(periods):
            if period_name not in trends_data[index_name]:
                continue
            
            subplot_idx += 1
            ax = plt.subplot(4, 3, subplot_idx, projection=proj)
            
            # Map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
            ax.set_global()
            
            # Get data
            period_data = trends_data[index_name][period_name]
            slopes = period_data['slopes'] * 10  # Convert to per decade
            significant = period_data['significant']
            
            # AGGRESSIVE MASKING - Multiple steps to prevent leakage
            print(f"  Masking {index_name} {period_name}...")
            
            # Step 1: Start with NaN everywhere
            slopes_masked = np.full_like(slopes, np.nan, dtype=np.float64)
            
            # Step 2: Only fill values where mask is True
            slopes_masked[mask] = slopes[mask]
            
            # Step 3: Explicitly ensure no values outside mask
            slopes_masked[~mask] = np.nan
            
            # Step 4: Additional buffer around mask boundaries to prevent interpolation
            if mask_type in ['land', 'ocean']:
                from scipy.ndimage import binary_erosion
                # Erode mask slightly to create a buffer zone
                mask_eroded = binary_erosion(mask, iterations=1)
                slopes_masked[~mask_eroded] = np.nan
                print(f"    Applied erosion buffer: {np.sum(mask_eroded)} pixels vs {np.sum(mask)} original")
            
            # Step 5: Final validation
            valid_outside_mask = np.sum(~np.isnan(slopes_masked[~mask]))
            valid_inside_mask = np.sum(~np.isnan(slopes_masked[mask]))
            print(f"    Final check: {valid_inside_mask} valid inside mask, {valid_outside_mask} outside (should be 0)")
            
            if valid_outside_mask > 0:
                print(f"    ERROR: Still {valid_outside_mask} values outside mask! Force-clearing...")
                slopes_masked[~mask] = np.nan
            
            # Plot with multiple fallback methods
            cmap = plt.cm.get_cmap(index_info['cmap'])
            
            # Method 1: Try masked array with pcolormesh (most reliable for masking)
            try:
                slopes_ma = np.ma.masked_invalid(slopes_masked)
                slopes_ma = np.ma.masked_where(~mask, slopes_ma)  # Double mask
                
                im = ax.pcolormesh(wdxxr_ds.longitude, wdxxr_ds.latitude, slopes_ma,
                                 cmap=cmap, transform=ccrs.PlateCarree(),
                                 vmin=vmin, vmax=vmax, shading='auto')
                print(f"    Used pcolormesh for {index_name} {period_name}")
                
            except Exception as e1:
                print(f"    pcolormesh failed: {e1}, trying contourf...")
                
                # Method 2: Try contourf with extended masking
                try:
                    # Create a version with extended NaN buffer for contourf
                    slopes_for_contour = slopes_masked.copy()
                    
                    # Add extra NaN buffer around boundaries
                    from scipy.ndimage import binary_dilation
                    mask_dilated = binary_dilation(~mask, iterations=2)  # Dilate the ocean/land boundaries
                    slopes_for_contour[mask_dilated] = np.nan
                    
                    im = ax.contourf(wdxxr_ds.longitude, wdxxr_ds.latitude, slopes_for_contour,
                                   levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                                   extend='both')
                    print(f"    Used contourf with buffer for {index_name} {period_name}")
                    
                except Exception as e2:
                    print(f"    contourf also failed: {e2}, using imshow...")
                    
                    # Method 3: Last resort - imshow
                    im = ax.imshow(slopes_masked, cmap=cmap, vmin=vmin, vmax=vmax,
                                 extent=[wdxxr_ds.longitude.min(), wdxxr_ds.longitude.max(),
                                        wdxxr_ds.latitude.min(), wdxxr_ds.latitude.max()],
                                 transform=ccrs.PlateCarree(), aspect='auto')
            
            # Add significance with proper masking
            if show_significance and np.any(significant & mask):
                if significance_method == 'contour':
                    add_significance_contours(ax, wdxxr_ds, significant, mask)
            
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
            
            # Add total pixel count
            # if len(valid_trends) > 0:
            #     stats_text += f'\n{len(valid_trends):,} pixels' if stats_text else f'{len(valid_trends):,} pixels'
            
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
    title_text = 'Trends in Chronological Precipitation Indices (WDXXR)\nMann-Kendall Analysis'
    if show_significance:
        title_text += '\n(Contours indicate statistical significance, p < 0.10)'
    
    # Add mask type to title
    if mask_type != 'both':
        title_text = f'{title_text}\n({mask_type.title()}-only analysis)'
    
    #fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.96)
    
    # Layout
    plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.05,
                       wspace=0.05, hspace=0)
    
    # Save
    filename = f'wdxxr_trends_{mask_type}'
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
    """Main function for WDXXR trends analysis."""
    parser = argparse.ArgumentParser(description='WDXXR chronological precipitation trends analysis')
    
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices',
                       help='Directory containing WDXXR index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
    parser.add_argument('--percentiles', nargs='+', type=int, default=[25, 50, 75, 90],
                       help='Percentiles to analyze (default: 25 50 75 90)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain (default: both)')
    parser.add_argument('--no-significance', action='store_true',
                       help='Hide statistical significance')
    parser.add_argument('--sig-method', choices=['contour', 'boundary'], default='contour',
                       help='Significance visualization method (default: contour)')
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes for parallel computation')
    parser.add_argument('--load-trends', type=str, default=None,
                       help='Load existing trends file instead of calculating')
    parser.add_argument('--calculate-only', action='store_true',
                       help='Calculate and save trends only, skip figure creation')
    parser.add_argument('--land-only', action='store_true',
                       help='Shortcut for --mask-type land')
    parser.add_argument('--ocean-only', action='store_true',
                       help='Shortcut for --mask-type ocean')
    parser.add_argument('--debug-mask', action='store_true',
                       help='Create debug figure showing land/ocean mask application')
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.land_only:
        args.mask_type = 'land'
    elif args.ocean_only:
        args.mask_type = 'ocean'
    
    print("="*80)
    print("WDXXR CHRONOLOGICAL PRECIPITATION TRENDS ANALYSIS")
    print("="*80)
    print(f"Input directory: {args.wd50r_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Percentiles: {args.percentiles}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Show significance: {not args.no_significance}")
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
            trends_data = load_wdxxr_trends_permanent(output_dir, filename=args.load_trends)
            if trends_data is None:
                raise ValueError(f"Could not load trends from {args.load_trends}")
            
            # Load dataset for coordinates (needed for plotting)
            wdxxr_ds = load_wdxxr_data_optimized(args.wd50r_dir, args.percentiles)
            
        else:
            # Load data
            print("Loading WDXXR data...")
            wdxxr_ds = load_wdxxr_data_optimized(args.wd50r_dir, args.percentiles)
            
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
            
            for percentile in args.percentiles:
                var_name = f'WD{percentile}R'
                
                if var_name not in wdxxr_ds.data_vars:
                    print(f"Warning: {var_name} not found in dataset")
                    continue
                
                print(f"\nProcessing {var_name}...")
                trends_data[var_name] = {}
                
                for period_name, (start_year, end_year) in periods.items():
                    period_ds = wdxxr_ds.sel(year=slice(start_year, end_year))
                    period_years = period_ds.year.values.astype(int)
                    
                    if len(period_years) < 4:
                        print(f"  Skipping {period_name}: insufficient data ({len(period_years)} years)")
                        continue
                    
                    print(f"  Calculating trends for {period_name} ({len(period_years)} years)...")
                    period_start = time.time()
                    
                    slopes, p_values, z_stats = calculate_trends_parallel(
                        period_ds[var_name], period_years, args.n_processes
                    )
                    
                    significant = p_values < 0.10
                    
                    trends_data[var_name][period_name] = {
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
                'years': sorted([int(y) for y in wdxxr_ds.year.values]),
                'percentiles': args.percentiles,
                'periods': {k: [int(v[0]), int(v[1])] for k, v in periods.items()},
                'shape': [len(wdxxr_ds.latitude), len(wdxxr_ds.longitude)],
                'lat_range': [float(wdxxr_ds.latitude.min()), float(wdxxr_ds.latitude.max())],
                'lon_range': [float(wdxxr_ds.longitude.min()), float(wdxxr_ds.longitude.max())]
            }
            
            trends_file = save_wdxxr_trends_permanent(trends_data, data_info, output_dir, args.mask_type)
            print(f"\nPermanent trends data saved: {trends_file}")
        
        if args.calculate_only:
            print("Calculation complete. Skipping figure creation.")
            return 0
        
        # Debug mask if requested
        if args.debug_mask:
            print("\nCreating mask debug visualization...")
            mask_debug = create_optimized_land_ocean_mask(wdxxr_ds, args.mask_type)
            save_mask_debug_figure(wdxxr_ds, mask_debug, args.mask_type, output_dir)
        
        # Create figure
        print("\nCreating WDXXR trends figure...")
        output_path = create_wdxxr_trends_figure(
            wdxxr_ds, trends_data, output_dir, args.mask_type,
            show_significance=not args.no_significance,
            significance_method=args.sig_method
        )
        
        print("\n" + "="*80)
        print("WDXXR ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print(f"Percentiles analyzed: {args.percentiles}")
        print("Analysis shows trends in precipitation clustering patterns:")
        print("- WD25R: Most concentrated events")
        print("- WD50R: Moderate concentration") 
        print("- WD75R: Broader patterns")
        print("- WD90R: Most distributed events")
        
        wdxxr_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())