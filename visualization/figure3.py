#!/usr/bin/env python3
"""
Create Publication Figure: WD50 vs WD50R Comparison Analysis

High-performance version with:
- Multi-year mean calculations for WD50 and WD50R indices
- Ratio and difference analysis with threshold-based filtering
- Permanent HDF5 data storage for repeated use
- Scientific-grade visualization and statistical analysis

Figure Layout: 2 rows × 2 columns
- (a) Mean WD50: Average wettest days for 50% precipitation (ETCCDI)
- (b) Mean WD50R: Average consecutive days for 50% precipitation (Chronological)
- (c) Ratio WD50/WD50R: Efficiency comparison (higher = more clustered)
- (d) Difference WD50R-WD50: Temporal clustering effect

Threshold Logic:
- Only calculate ratio and difference where Mean WD50 > threshold
- Pixels below threshold show NaN (masked out)

Usage:
python figure3.py --etccdi-dir data/processed/etccdi_indices --wd50r-dir data/processed/wd50r_indices --output-dir figures/
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
from numba import jit, prange
import glob

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
def calculate_ratios_differences_vectorized(wd50_mean, wd50r_mean, threshold):
    """
    Vectorized calculation of ratios and differences with threshold logic.
    
    Args:
        wd50_mean: 2D array of mean WD50 values
        wd50r_mean: 2D array of mean WD50R values  
        threshold: Minimum WD50 value to calculate ratio/difference
    
    Returns:
        ratios: WD50/WD50R ratios (NaN where WD50 <= threshold)
        differences: WD50R-WD50 differences (NaN where WD50 <= threshold)
    """
    nlat, nlon = wd50_mean.shape
    ratios = np.full((nlat, nlon), np.nan)
    differences = np.full((nlat, nlon), np.nan)
    
    for i in prange(nlat):
        for j in range(nlon):
            wd50_val = wd50_mean[i, j]
            wd50r_val = wd50r_mean[i, j]
            
            # Only calculate if WD50 > threshold and both values are valid
            if (not np.isnan(wd50_val) and not np.isnan(wd50r_val) and 
                wd50_val > threshold):
                
                # Ratio: WD50/WD50R (efficiency metric)
                if wd50r_val > 0:
                    ratios[i, j] = wd50_val / wd50r_val
                
                # Difference: WD50R - WD50 (clustering effect)
                differences[i, j] = wd50r_val - wd50_val
    
    return ratios, differences

def process_spatial_chunk_means(args):
    """Process a chunk of pixels for multiprocessing mean calculations."""
    wd50_chunk, wd50r_chunk, i_start, i_end, j_start, j_end = args
    
    wd50_mean = np.full((i_end - i_start, j_end - j_start), np.nan)
    wd50r_mean = np.full((i_end - i_start, j_end - j_start), np.nan)
    
    for i in range(i_end - i_start):
        for j in range(j_end - j_start):
            # Calculate means across years, ignoring NaN values
            wd50_pixel = wd50_chunk[:, i, j]
            wd50r_pixel = wd50r_chunk[:, i, j]
            
            # Mean WD50 (from ETCCDI)
            valid_wd50 = wd50_pixel[~np.isnan(wd50_pixel)]
            if len(valid_wd50) > 0:
                wd50_mean[i, j] = np.mean(valid_wd50)
            
            # Mean WD50R (from chronological analysis)
            valid_wd50r = wd50r_pixel[~np.isnan(wd50r_pixel)]
            if len(valid_wd50r) > 0:
                wd50r_mean[i, j] = np.mean(valid_wd50r)
    
    return wd50_mean, wd50r_mean, i_start, i_end, j_start, j_end

def calculate_means_parallel(wd50_data, wd50r_data, n_processes=None, chunk_size=50):
    """
    Calculate multi-year means using multiprocessing.
    
    Args:
        wd50_data: xarray DataArray with time dimension (WD50)
        wd50r_data: xarray DataArray with time dimension (WD50R) 
        n_processes: number of processes
        chunk_size: size of spatial chunks
    
    Returns:
        wd50_mean: mean WD50 values
        wd50r_mean: mean WD50R values
    """
    if n_processes is None:
        n_processes = min(192, mp.cpu_count())
    
    print(f"Using {n_processes} processes for parallel mean calculation")
    
    # Get data dimensions - ensure they match
    wd50_3d = wd50_data.values
    wd50r_3d = wd50r_data.values
    
    if wd50_3d.shape != wd50r_3d.shape:
        raise ValueError(f"WD50 and WD50R data shapes don't match: {wd50_3d.shape} vs {wd50r_3d.shape}")
    
    nt, nlat, nlon = wd50_3d.shape
    print(f"Processing {nt} years, {nlat}x{nlon} grid")
    
    # Initialize output arrays
    wd50_mean = np.full((nlat, nlon), np.nan)
    wd50r_mean = np.full((nlat, nlon), np.nan)
    
    # Create chunks for parallel processing
    chunks = []
    for i in range(0, nlat, chunk_size):
        for j in range(0, nlon, chunk_size):
            i_end = min(i + chunk_size, nlat)
            j_end = min(j + chunk_size, nlon)
            
            # Extract data chunks
            wd50_chunk = wd50_3d[:, i:i_end, j:j_end]
            wd50r_chunk = wd50r_3d[:, i:i_end, j:j_end]
            chunks.append((wd50_chunk, wd50r_chunk, i, i_end, j, j_end))
    
    print(f"Processing {len(chunks)} spatial chunks")
    
    # Process chunks in parallel
    with Pool(n_processes) as pool:
        results = pool.map(process_spatial_chunk_means, chunks)
    
    # Combine results
    for wd50_chunk_mean, wd50r_chunk_mean, i_start, i_end, j_start, j_end in results:
        wd50_mean[i_start:i_end, j_start:j_end] = wd50_chunk_mean
        wd50r_mean[i_start:i_end, j_start:j_end] = wd50r_chunk_mean
    
    return wd50_mean, wd50r_mean

def create_optimized_land_ocean_mask(ds, mask_type='both', validate=True):
    """
    Optimized land/ocean mask creation with validation.
    (Same as figure2.py implementation)
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
    
    # Enhanced fallback approach
    print("  Using enhanced geometric land/ocean approximation...")
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    land_mask = np.zeros_like(lat_grid, dtype=bool)
    
    # Continental boundaries (simplified but effective)
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
    Load WD50 data from ETCCDI indices.
    (Adapted from figure2.py implementation)
    """
    etccdi_dir = Path(etccdi_dir)
    
    print(f"Loading ETCCDI WD50 data from: {etccdi_dir.absolute()}")
    
    if years is None:
        # Find ETCCDI files
        etccdi_files = list(etccdi_dir.glob('etccdi_precipitation_indices_*.nc'))
        
        if not etccdi_files:
            # Try alternative patterns
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
    
    # Load WD50 data
    wd50_data_list = []
    valid_years = []
    
    for year in years:
        file_path = etccdi_dir / f'etccdi_precipitation_indices_{year}.nc'
        
        if file_path.exists():
            try:
                with xr.open_dataset(file_path) as ds:
                    if 'WD50' in ds.data_vars:
                        # Handle timedelta conversion
                        if 'timedelta' in str(ds.WD50.dtype):
                            wd50_values = ds.WD50 / np.timedelta64(1, 'D')
                            wd50_values = wd50_values.astype(np.float64)
                        else:
                            wd50_values = ds.WD50
                        
                        wd50_data_list.append(wd50_values.values)
                        valid_years.append(year)
                        
                        # Get coordinates from first file
                        if len(valid_years) == 1:
                            lat_coords = ds.latitude.values
                            lon_coords = ds.longitude.values
                            
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    if not wd50_data_list:
        raise ValueError("No valid WD50 data found")
    
    # Combine into dataset
    wd50_array = np.stack(wd50_data_list, axis=0)
    
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    wd50_da = xr.DataArray(
        wd50_array,
        dims=['year', 'latitude', 'longitude'],
        coords=coords,
        name='WD50'
    )
    
    wd50_ds = xr.Dataset({'WD50': wd50_da}, coords=coords)
    
    print(f"Successfully loaded WD50: {len(valid_years)} years, {wd50_array.shape[1]}x{wd50_array.shape[2]} grid")
    return wd50_ds

def load_wd50r_data_optimized(wd50r_dir, years=None, percentile=50):
    """
    Load WD50R data from chronological precipitation indices.
    """
    wd50r_dir = Path(wd50r_dir)
    
    print(f"Loading WD50R data from: {wd50r_dir.absolute()}")
    
    if years is None:
        # Find WD50R files
        wd50r_files = list(wd50r_dir.glob(f'wd50r_indices_*_P*{percentile}*.nc'))
        
        if not wd50r_files:
            # Try more general pattern
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
    
    # Load WD50R data
    wd50r_data_list = []
    valid_years = []
    
    for year in years:
        # Find file for this year
        year_files = list(wd50r_dir.glob(f'wd50r_indices_{year}_*.nc'))
        
        if year_files:
            file_path = year_files[0]  # Take first match
            
            try:
                with xr.open_dataset(file_path) as ds:
                    if f'WD{percentile}R' in ds.data_vars:
                        wd50r_var = ds[f'WD{percentile}R']
                        
                        # Debug: Check data type and sample values
                        if len(valid_years) == 0:  # First file
                            print(f"  WD{percentile}R dtype: {wd50r_var.dtype}")
                            sample_vals = wd50r_var.values.flatten()
                            valid_sample = sample_vals[~np.isnan(sample_vals)][:5]
                            print(f"  Sample values: {valid_sample}")
                        
                        # Handle timedelta conversion (similar to WD50 in ETCCDI)
                        if 'timedelta' in str(wd50r_var.dtype):
                            #print(f"  Converting WD{percentile}R from timedelta to days")
                            wd50r_values = wd50r_var / np.timedelta64(1, 'D')
                            wd50r_values = wd50r_values.astype(np.float64)
                        elif 'datetime' in str(wd50r_var.dtype):
                            #print(f"  Converting WD{percentile}R from datetime to days (unusual)")
                            # This shouldn't happen, but handle just in case
                            wd50r_values = wd50r_var.astype(np.float64)
                        else:
                            # Check if values are suspiciously large (likely nanoseconds)
                            sample_vals = wd50r_var.values.flatten()
                            valid_sample = sample_vals[~np.isnan(sample_vals)]
                            
                            if len(valid_sample) > 0 and np.mean(valid_sample) > 1e10:
                                print(f"  Detected large values (likely nanoseconds), converting to days")
                                print(f"  Sample large value: {valid_sample[0]:.0f}")
                                # Convert from nanoseconds to days
                                wd50r_values = wd50r_var / 1e9 / 86400  # ns -> seconds -> days
                                wd50r_values = wd50r_values.astype(np.float64)
                            elif len(valid_sample) > 0 and np.mean(valid_sample) > 1e6:
                                print(f"  Detected large values (likely microseconds), converting to days")
                                print(f"  Sample large value: {valid_sample[0]:.0f}")
                                # Convert from microseconds to days
                                wd50r_values = wd50r_var / 1e6 / 86400  # μs -> seconds -> days
                                wd50r_values = wd50r_values.astype(np.float64)
                            else:
                                wd50r_values = wd50r_var.astype(np.float64)
                        
                        # Debug: Check converted values
                        if len(valid_years) == 0:  # First file
                            conv_sample = wd50r_values.values.flatten()
                            valid_conv = conv_sample[~np.isnan(conv_sample)][:5]
                            print(f"  Converted sample values: {valid_conv}")
                        
                        wd50r_data_list.append(wd50r_values.values)
                        valid_years.append(year)
                        
                        # Get coordinates from first file
                        if len(valid_years) == 1:
                            lat_coords = ds.latitude.values
                            lon_coords = ds.longitude.values
                            
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
    
    if not wd50r_data_list:
        raise ValueError(f"No valid WD{percentile}R data found")
    
    # Combine into dataset
    wd50r_array = np.stack(wd50r_data_list, axis=0)
    
    coords = {
        'year': valid_years,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    wd50r_da = xr.DataArray(
        wd50r_array,
        dims=['year', 'latitude', 'longitude'],
        coords=coords,
        name='WD50R'
    )
    
    wd50r_ds = xr.Dataset({'WD50R': wd50r_da}, coords=coords)
    
    # Final validation of loaded data
    sample_data = wd50r_ds.WD50R.values.flatten()
    valid_final = sample_data[~np.isnan(sample_data)]
    
    if len(valid_final) > 0:
        print(f"Successfully loaded WD50R: {len(valid_years)} years, {wd50r_array.shape[1]}x{wd50r_array.shape[2]} grid")
        print(f"Final data range: {np.min(valid_final):.2f} to {np.max(valid_final):.2f} days")
        print(f"Final data mean: {np.mean(valid_final):.2f} ± {np.std(valid_final):.2f} days")
    
    return wd50r_ds

def save_comparison_data_permanent(comparison_data, data_info, output_dir, mask_type):
    """
    Save WD50 vs WD50R comparison data to permanent HDF5 format.
    """
    # Create unique filename
    years_str = f"{data_info['years'][0]}-{data_info['years'][-1]}"
    shape_str = f"{data_info['shape'][0]}x{data_info['shape'][1]}"
    threshold_str = f"thresh{data_info['threshold']:.0f}"
    filename = f"wd50_comparison_{years_str}_{shape_str}_{threshold_str}_{mask_type}.h5"
    
    comparison_file = output_dir / filename
    
    print(f"Saving permanent comparison data to: {comparison_file}")
    
    with h5py.File(comparison_file, 'w') as f:
        # Save metadata
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['mask_type'] = mask_type
        metadata_grp.attrs['threshold'] = data_info['threshold']
        metadata_grp.attrs['created_date'] = pd.Timestamp.now().isoformat()
        metadata_grp.attrs['years_range'] = f"{data_info['years'][0]}-{data_info['years'][-1]}"
        metadata_grp.attrs['grid_shape'] = data_info['shape']
        metadata_grp.attrs['lat_range'] = data_info['lat_range']
        metadata_grp.attrs['lon_range'] = data_info['lon_range']
        metadata_grp.create_dataset('years', data=np.array(data_info['years']))
        
        # Save comparison data
        comp_grp = f.create_group('comparison')
        
        for var_name, var_data in comparison_data.items():
            comp_grp.create_dataset(var_name, data=var_data,
                                  compression='gzip', compression_opts=9)
        
        # Statistics
        stats_grp = f.create_group('statistics')
        for var_name, var_data in comparison_data.items():
            valid_data = var_data[~np.isnan(var_data)]
            if len(valid_data) > 0:
                stats_grp.attrs[f'{var_name}_mean'] = np.mean(valid_data)
                stats_grp.attrs[f'{var_name}_std'] = np.std(valid_data)
                stats_grp.attrs[f'{var_name}_min'] = np.min(valid_data)
                stats_grp.attrs[f'{var_name}_max'] = np.max(valid_data)
                stats_grp.attrs[f'{var_name}_valid_pixels'] = len(valid_data)
    
    print(f"Comparison data permanently saved: {comparison_file.stat().st_size / (1024**2):.1f} MB")
    return comparison_file

def load_comparison_data_permanent(output_dir, mask_type=None, filename=None):
    """
    Load permanent comparison data from HDF5 format.
    """
    if filename:
        comparison_file = Path(filename)
    else:
        pattern = f"wd50_comparison_*_{mask_type}.h5" if mask_type else "wd50_comparison_*.h5"
        comparison_files = list(output_dir.glob(pattern))
        
        if not comparison_files:
            return None
        
        comparison_file = max(comparison_files, key=lambda x: x.stat().st_mtime)
    
    if not comparison_file.exists():
        return None
    
    print(f"Loading permanent comparison data from: {comparison_file}")
    
    with h5py.File(comparison_file, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        
        # Load comparison data
        comparison_data = {}
        comp_grp = f['comparison']
        for var_name in comp_grp.keys():
            comparison_data[var_name] = comp_grp[var_name][:]
        
        # Load statistics
        stats = dict(f['statistics'].attrs) if 'statistics' in f else {}
    
    return comparison_data, metadata, stats

def create_wd50_comparison_figure(comparison_data, lat_coords, lon_coords, output_dir, 
                                 mask_type='both', threshold=0.0):
    """
    Create WD50 vs WD50R comparison figure.
    """
    # Create mask
    dummy_ds = xr.Dataset(coords={'latitude': lat_coords, 'longitude': lon_coords})
    mask = create_optimized_land_ocean_mask(dummy_ds, mask_type)
    
    # Define subplot information
    subplots_info = {
        'wd50_mean': {
            'title': '(a) Mean WD50',
            'data_key': 'wd50_mean',
            'units': 'days',
            'cmap': 'viridis',
            'description': 'Average wettest days for 50% precipitation (ETCCDI)'
        },
        'wd50r_mean': {
            'title': '(b) Mean WD50R', 
            'data_key': 'wd50r_mean',
            'units': 'days',
            'cmap': 'viridis',
            'description': 'Average consecutive days for 50% precipitation'
        },
        'ratio': {
            'title': '(c) Ratio WD50/WD50R',
            'data_key': 'ratio',
            'units': 'dimensionless',
            'cmap': 'RdYlBu_r',
            'description': f'Efficiency ratio (calculated where WD50 > {threshold} days)'
        },
        'difference': {
            'title': '(d) Difference WD50R-WD50',
            'data_key': 'difference', 
            'units': 'days',
            'cmap': 'RdBu_r',
            'description': f'Clustering effect (calculated where WD50 > {threshold} days)'
        }
    }
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    proj = ccrs.Robinson(central_longitude=0)
    
    for idx, (subplot_key, subplot_info) in enumerate(subplots_info.items()):
        ax = plt.subplot(2, 2, idx + 1, projection=proj)
        
        # Map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax.set_global()
        
        # Get data
        data = comparison_data[subplot_info['data_key']]
        
        # Apply mask
        data_masked = data.copy()
        data_masked[~mask] = np.nan
        
        # Calculate statistics for valid data
        valid_data = data_masked[~np.isnan(data_masked)]
        
        if len(valid_data) == 0:
            ax.set_title(f"{subplot_info['title']}\n(No valid data)", fontsize=14, fontweight='bold')
            continue
        
        # Calculate actual data range
        actual_min = np.min(valid_data)
        actual_max = np.max(valid_data)
        
        # Determine color range
        if subplot_key in ['ratio', 'difference']:
            # Symmetric colorbars for ratio and difference
            if subplot_key == 'ratio':
                # Ratio around 1.0
                vmax_abs = max(1.0, np.percentile(np.abs(valid_data - 1.0), 95))
                vmin, vmax = max(0.1, 1.0 - vmax_abs), 1.0 + vmax_abs
                percentile_range = "5–95"
            else:
                # Difference symmetric around 0
                vmax_abs = np.percentile(np.abs(valid_data), 95)
                vmin, vmax = -vmax_abs, vmax_abs
                percentile_range = "5–95"
        else:
            # Mean values - regular range
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
            percentile_range = "2–98"
        
        # Create levels
        levels = np.linspace(vmin, vmax, 20)
        
        # Plot data
        cmap = plt.cm.get_cmap(subplot_info['cmap'])
        im = ax.contourf(lon_coords, lat_coords, data_masked,
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree(),
                        extend='both')
        
        # Add colorbar with comprehensive label
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        dummy_im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(dummy_im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, extend='both')
        
        # Comprehensive colorbar label
        if subplot_key in ['wd50_mean', 'wd50r_mean']:
            cbar_label = f"{subplot_info['units']}\nColor: {percentile_range} percentile\nActual: {actual_min:.1f} to {actual_max:.1f}"
        else:
            cbar_label = f"{subplot_info['units']}\nColor: {percentile_range} percentile\nActual: {actual_min:.2f} to {actual_max:.2f}"
        cbar.set_label(cbar_label, fontsize=9, linespacing=1.2)
        
        # Set ticks strictly within [vmin, vmax]
        tick_locs = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(tick_locs)
        if subplot_key in ['wd50_mean', 'wd50r_mean']:
            cbar.set_ticklabels([f'{val:.1f}' for val in tick_locs])
        else:
            cbar.set_ticklabels([f'{val:.2f}' for val in tick_locs])
        cbar.ax.tick_params(labelsize=9)
        
        # Statistics text
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        valid_pct = len(valid_data) / np.sum(mask) * 100
        
        stats_text = f"Mean: {mean_val:.1f} ± {std_val:.1f}\n{valid_pct:.1f}% valid"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
               verticalalignment='top', fontsize=10)
        
        # Title
        ax.set_title(subplot_info['title'], fontsize=14, fontweight='bold', pad=10)
    
    # Main title
    threshold_text = f"(Ratio/Difference calculated where Mean WD50 > {threshold} days)" if threshold > 0 else ""
    title_text = f'WD50 vs WD50R Comparison Analysis\n{threshold_text}'
    
    fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.95)
    
    # Layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save
    threshold_str = f"_thresh{threshold:.0f}" if threshold > 0 else ""
    filename = f'wd50_comparison_{mask_type}{threshold_str}.png'
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    return output_path

def main():
    """Main function for WD50 vs WD50R comparison analysis."""
    parser = argparse.ArgumentParser(description='WD50 vs WD50R comparison analysis and visualization')
    
    # Data directories
    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI index files (for WD50)')
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices', 
                       help='Directory containing WD50R index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
    
    # Analysis parameters
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain (default: both)')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Minimum WD50 value to calculate ratio/difference (default: 0.0)')
    
    # Processing parameters  
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes for parallel computation (default: auto)')
    parser.add_argument('--load-comparison', type=str, default=None,
                       help='Load existing comparison file instead of calculating')
    parser.add_argument('--calculate-only', action='store_true',
                       help='Calculate and save comparison data only, skip figure creation')
    
    # Shortcuts
    parser.add_argument('--land-only', action='store_true',
                       help='Shortcut for --mask-type land')
    parser.add_argument('--ocean-only', action='store_true',
                       help='Shortcut for --mask-type ocean')
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.land_only:
        args.mask_type = 'land'
    elif args.ocean_only:
        args.mask_type = 'ocean'
    
    print("="*80)
    print("WD50 vs WD50R COMPARISON ANALYSIS")
    print("="*80)
    print(f"ETCCDI directory: {args.etccdi_dir}")
    print(f"WD50R directory: {args.wd50r_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"WD50 threshold: {args.threshold} days")
    print(f"Processes: {args.n_processes if args.n_processes else 'auto'}")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load or calculate comparison data
        if args.load_comparison:
            print(f"Loading existing comparison data from: {args.load_comparison}")
            result = load_comparison_data_permanent(output_dir, filename=args.load_comparison)
            
            if result is None:
                raise ValueError(f"Could not load comparison data from {args.load_comparison}")
            
            comparison_data, metadata, stats = result
            
            # Load coordinates (need one dataset for coordinates)
            etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)
            lat_coords = etccdi_ds.latitude.values
            lon_coords = etccdi_ds.longitude.values
            
        else:
            # Load data
            print("Loading WD50 data from ETCCDI indices...")
            etccdi_ds = load_etccdi_data_optimized(args.etccdi_dir)
            
            print("Loading WD50R data from chronological indices...")
            wd50r_ds = load_wd50r_data_optimized(args.wd50r_dir)
            
            # Find common years and align datasets
            common_years = sorted(set(etccdi_ds.year.values) & set(wd50r_ds.year.values))
            if not common_years:
                raise ValueError("No common years found between WD50 and WD50R datasets")
            
            print(f"Common years: {len(common_years)} years ({common_years[0]}-{common_years[-1]})")
            
            # Align datasets to common years and ensure same spatial grid
            etccdi_aligned = etccdi_ds.sel(year=common_years)
            wd50r_aligned = wd50r_ds.sel(year=common_years)
            
            # Ensure same spatial coordinates
            if not np.array_equal(etccdi_aligned.latitude.values, wd50r_aligned.latitude.values):
                raise ValueError("WD50 and WD50R datasets have different latitude coordinates")
            if not np.array_equal(etccdi_aligned.longitude.values, wd50r_aligned.longitude.values):
                raise ValueError("WD50 and WD50R datasets have different longitude coordinates")
            
            lat_coords = etccdi_aligned.latitude.values
            lon_coords = etccdi_aligned.longitude.values
            
            # Calculate multi-year means
            print("Calculating multi-year means...")
            start_time = time.time()
            
            wd50_mean, wd50r_mean = calculate_means_parallel(
                etccdi_aligned.WD50, wd50r_aligned.WD50R, args.n_processes
            )
            
            # Calculate ratios and differences with threshold logic
            print(f"Calculating ratios and differences (threshold: {args.threshold} days)...")
            ratios, differences = calculate_ratios_differences_vectorized(
                wd50_mean, wd50r_mean, args.threshold
            )
            
            calculation_time = time.time() - start_time
            print(f"Total calculation time: {calculation_time:.1f}s")
            
            # Prepare comparison data
            comparison_data = {
                'wd50_mean': wd50_mean,
                'wd50r_mean': wd50r_mean,
                'ratio': ratios,
                'difference': differences
            }
            
            # Statistics
            mask = create_optimized_land_ocean_mask(etccdi_aligned, args.mask_type)
            
            for var_name, data in comparison_data.items():
                valid_data = data[mask & ~np.isnan(data)]
                if len(valid_data) > 0:
                    print(f"{var_name}: {len(valid_data)} valid pixels, "
                          f"mean={np.mean(valid_data):.2f}, std={np.std(valid_data):.2f}")
            
            # Save permanent data
            data_info = {
                'years': common_years,
                'threshold': args.threshold,
                'shape': [len(lat_coords), len(lon_coords)],
                'lat_range': [float(np.min(lat_coords)), float(np.max(lat_coords))],
                'lon_range': [float(np.min(lon_coords)), float(np.max(lon_coords))]
            }
            
            comparison_file = save_comparison_data_permanent(
                comparison_data, data_info, output_dir, args.mask_type
            )
            print(f"Permanent comparison data saved: {comparison_file}")
        
        if args.calculate_only:
            print("Calculation complete. Skipping figure creation.")
            return 0
        
        # Create figure
        print("Creating WD50 vs WD50R comparison figure...")
        output_path = create_wd50_comparison_figure(
            comparison_data, lat_coords, lon_coords, output_dir,
            args.mask_type, args.threshold
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print("Analysis components:")
        print("- Mean WD50: Average wettest days for 50% precipitation")
        print("- Mean WD50R: Average consecutive days for 50% precipitation") 
        print("- Ratio: Efficiency comparison (WD50/WD50R)")
        print("- Difference: Clustering effect (WD50R-WD50)")
        print(f"- Threshold applied: {args.threshold} days")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())