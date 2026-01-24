#!/usr/bin/env python3
"""
Create Publication Figure: Trends in Multi-Threshold Precipitation Concentration Indices (WDXX)

High-performance version with:
- Vectorized Mann-Kendall trend analysis for WD25, WD50, WD75
- Multiprocessing for 192-core systems
- Permanent data storage (no caching)
- Scientific-grade statistical testing

Figure Layout: 3 rows × 3 columns
- Rows: WD25, WD50, WD75 (multi-threshold concentration)
- Columns: 1980-2024, 1980-1999 (pre-2000), 2000-2024 (post-2000)

Usage:
python figure7.py --enhanced-dir data/processed/enhanced_concentration_indices --output-dir figures/
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
            
            # Clean up mask with additional robustness
            from scipy.ndimage import binary_opening, binary_closing
            if land_mask.sum() > 0:
                land_mask = binary_closing(land_mask, iterations=1)
                land_mask = binary_opening(land_mask, iterations=1)
                # Additional cleanup: remove isolated pixels
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
                                #print(f"  Converting {var} from timedelta to days for year {year}")
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

def save_wdxx_trends_permanent(trends_data, data_info, output_dir, mask_type):
    """
    Save WDXX trends data to permanent HDF5 format.
    """
    # Create unique filename
    years_str = f"{data_info['years'][0]}-{data_info['years'][-1]}"
    shape_str = f"{data_info['shape'][0]}x{data_info['shape'][1]}"
    filename = f"wdxx_trends_{years_str}_{shape_str}_{mask_type}.h5"
    
    trends_file = output_dir / filename
    
    print(f"Saving permanent WDXX trends data to: {trends_file}")
    
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
        
        # Handle variables with robust encoding
        try:
            variables_data = np.array([var.encode('utf-8') for var in data_info['variables']], dtype='S')
        except:
            variables_data = np.array(data_info['variables'], dtype='S')
        
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
    
    print(f"WDXX trends data permanently saved: {trends_file.stat().st_size / (1024**3):.2f} GB")
    return trends_file

def load_wdxx_trends_permanent(output_dir, mask_type=None, filename=None):
    """
    Load permanent WDXX trends data from HDF5 format.
    """
    if filename:
        trends_file = Path(filename)
    else:
        pattern = f"wdxx_trends_*_{mask_type}.h5" if mask_type else "wdxx_trends_*.h5"
        trends_files = list(output_dir.glob(pattern))
        
        if not trends_files:
            return None
        
        trends_file = max(trends_files, key=lambda x: x.stat().st_mtime)
    
    if not trends_file.exists():
        return None
    
    print(f"Loading permanent WDXX trends data from: {trends_file}")
    
    trends_data = {}
    
    with h5py.File(trends_file, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        
        # Handle variables with robust loading
        try:
            if 'variables' in f['metadata']:
                variables_data = f['metadata/variables'][:]
                if hasattr(variables_data[0], 'decode'):
                    variables = [s.decode('utf-8') for s in variables_data]
                else:
                    variables = [str(s) for s in variables_data]
            else:
                raise KeyError("Variables dataset not found")
        except (KeyError, TypeError) as e:
            print(f"  Variables dataset not found, scanning for WD variables: {e}")
            # Scan for WD variables
            possible_vars = ['WD25', 'WD50', 'WD75']
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
    Add significance contours.
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

def create_wdxx_trends_figure(enhanced_ds, trends_data, output_dir, mask_type='both',
                             show_significance=True, significance_method='contour'):
    """
    Create WDXX trends figure (3 rows × 3 columns).
    """
    # Create mask
    mask = create_optimized_land_ocean_mask(enhanced_ds, mask_type)
    
    # Define indices and their properties
    indices_info = {
        'WD25': {'units': 'days/decade', 'title': 'WD25', 'cmap': 'RdYlBu', 'symmetric': True},
        'WD50': {'units': 'days/decade', 'title': 'WD50', 'cmap': 'RdYlBu', 'symmetric': True},
        'WD75': {'units': 'days/decade', 'title': 'WD75', 'cmap': 'RdYlBu', 'symmetric': True}
    }
    
    periods = ['1980-2024', '1980-1999', '2000-2024']
    
    # Create figure
    fig = plt.figure(figsize=(15, 9))  
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
        
        # Determine colorbar range (symmetric)
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
            ax = plt.subplot(3, 3, subplot_idx, projection=proj)
            
            # Map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
            ax.set_global()
            
            # Get data
            period_data = trends_data[index_name][period_name]
            slopes = period_data['slopes'] * 10  # Convert to per decade
            significant = period_data['significant']
            
            # Apply robust mask to prevent color leakage
            slopes_masked = slopes.copy()

            # Create a more robust mask by adding morphological operations
            try:
                from scipy.ndimage import binary_erosion, binary_dilation
                # Dilate the mask slightly to create a buffer zone
                if mask.sum() > 0:
                    mask_eroded = binary_erosion(mask, iterations=1)
                    # Set buffer zone to NaN to prevent interpolation artifacts
                    slopes_masked[mask_eroded & ~mask] = np.nan
            except ImportError:
                print("  scipy not available for morphological operations")

            # Apply main mask
            slopes_masked[~mask] = np.nan

            # Additional check: ensure all non-masked values are finite
            slopes_masked[~np.isfinite(slopes_masked) & mask] = np.nan

            # Plot trends using pcolormesh for better mask handling (no interpolation artifacts)
            cmap = plt.cm.get_cmap(index_info['cmap'])

            # Create coordinate arrays for pcolormesh (need to be 1 element larger than data)
            lon_vals = enhanced_ds.longitude.values
            lat_vals = enhanced_ds.latitude.values

            # Create boundaries for flat shading (corners of grid cells)
            lon_bounds = np.concatenate([lon_vals - (lon_vals[1] - lon_vals[0])/2,
                                       [lon_vals[-1] + (lon_vals[1] - lon_vals[0])/2]])
            lat_bounds = np.concatenate([lat_vals - (lat_vals[1] - lat_vals[0])/2,
                                       [lat_vals[-1] + (lat_vals[1] - lat_vals[0])/2]])

            # Use pcolormesh with flat shading (default) which requires boundary coordinates
            im = ax.pcolormesh(lon_bounds, lat_bounds, slopes_masked,
                             cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree(), shading='flat')
            
            # Add significance
            if show_significance:
                if significance_method == 'contour':
                    add_significance_contours(ax, enhanced_ds, significant, mask)
            
            # Statistics
            sig_percentage = np.sum(significant & mask) / np.sum(mask) * 100 if np.sum(mask) > 0 else 0
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(period_name, fontsize=14, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ylabel = f"({chr(97 + row_idx)}) {index_info['title']}"
                ax.text(-0.15, 0.5, ylabel, transform=ax.transAxes, rotation=90,
                       ha='center', va='center', fontweight='bold', fontsize=14)
            
            # Add statistics
            if show_significance:
                ax.text(0.02, 0.02, f'{sig_percentage:.1f}% sig.',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add colorbar
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
    title_text = 'Trends in Multi-Threshold Precipitation Concentration Indices (WDXX)\nMann-Kendall Analysis'
    if show_significance:
        title_text += '\n(Contours indicate statistical significance, p < 0.10)'
    
    #fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.95)
    
    # Layout
    plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.05,
                       wspace=0.05, hspace=-0.05)
    
    # Save
    filename = f'wdxx_trends_{mask_type}'
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
    """Main function for WDXX trends analysis."""
    parser = argparse.ArgumentParser(description='WDXX multi-threshold precipitation trends analysis')
    
    parser.add_argument('--enhanced-dir', default='data/processed/enhanced_concentration_indices',
                       help='Directory containing enhanced concentration index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
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
    
    args = parser.parse_args()
    
    # Handle shortcuts
    if args.land_only:
        args.mask_type = 'land'
    elif args.ocean_only:
        args.mask_type = 'ocean'
    
    print("="*80)
    print("WDXX MULTI-THRESHOLD PRECIPITATION TRENDS ANALYSIS")
    print("="*80)
    print(f"Input directory: {args.enhanced_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Show significance: {not args.no_significance}")
    print(f"Processes: {args.n_processes if args.n_processes else 'auto'}")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load or calculate trends
        if args.load_trends:
            print(f"Loading existing trends from: {args.load_trends}")
            trends_data = load_wdxx_trends_permanent(output_dir, filename=args.load_trends)
            if trends_data is None:
                raise ValueError(f"Could not load trends from {args.load_trends}")
            
            # Load dataset for coordinates (needed for plotting)
            enhanced_ds = load_enhanced_concentration_data_optimized(args.enhanced_dir)
            
        else:
            # Load data
            print("Loading enhanced concentration data...")
            enhanced_ds = load_enhanced_concentration_data_optimized(args.enhanced_dir)
            
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
            
            for index_name in ['WD25', 'WD50', 'WD75']:
                if index_name not in enhanced_ds.data_vars:
                    print(f"Warning: {index_name} not found in dataset")
                    continue
                
                print(f"\nProcessing {index_name}...")
                trends_data[index_name] = {}
                
                for period_name, (start_year, end_year) in periods.items():
                    period_ds = enhanced_ds.sel(year=slice(start_year, end_year))
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
                'years': sorted([int(y) for y in enhanced_ds.year.values]),
                'variables': ['WD25', 'WD50', 'WD75'],
                'periods': {k: [int(v[0]), int(v[1])] for k, v in periods.items()},
                'shape': [len(enhanced_ds.latitude), len(enhanced_ds.longitude)],
                'lat_range': [float(enhanced_ds.latitude.min()), float(enhanced_ds.latitude.max())],
                'lon_range': [float(enhanced_ds.longitude.min()), float(enhanced_ds.longitude.max())]
            }
            
            trends_file = save_wdxx_trends_permanent(trends_data, data_info, output_dir, args.mask_type)
            print(f"\nPermanent trends data saved: {trends_file}")
        
        if args.calculate_only:
            print("Calculation complete. Skipping figure creation.")
            return 0
        
        # Create figure
        print("\nCreating WDXX trends figure...")
        output_path = create_wdxx_trends_figure(
            enhanced_ds, trends_data, output_dir, args.mask_type,
            show_significance=not args.no_significance,
            significance_method=args.sig_method
        )
        
        print("\n" + "="*80)
        print("WDXX ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print("Analysis shows trends in multi-threshold precipitation concentration:")
        print("- WD25: Extreme concentration (25% threshold)")
        print("- WD50: Moderate concentration (50% threshold)")
        print("- WD75: Distributed patterns (75% threshold)")
        
        enhanced_ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())