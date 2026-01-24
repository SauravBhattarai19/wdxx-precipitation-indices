#!/usr/bin/env python3
"""
Create Publication Figure: Multi-Threshold WD vs WDXXR Comparison Analysis

High-performance version with:
- Multi-year mean calculations for WDXX and WDXXR indices across all thresholds
- Ratio and difference analysis for WD25/25R, WD50/50R, WD75/75R
- Permanent HDF5 data storage for repeated use
- Scientific-grade visualization and statistical analysis

Figure Layout: 3 rows × 2 columns
- (a,b) WD25 vs WD25R: Ratio and Difference for 25% threshold
- (c,d) WD50 vs WD50R: Ratio and Difference for 50% threshold  
- (e,f) WD75 vs WD75R: Ratio and Difference for 75% threshold

Usage:
python figure6.py --etccdi-dir data/processed/etccdi_indices --enhanced-dir data/processed/enhanced_concentration_indices --wd50r-dir data/processed/wd50r_indices --output-dir figures/
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
    'axes.linewidth': 1,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 16,
    'figure.titlesize': 20,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

@jit(nopython=True, parallel=True)
def calculate_ratios_differences_vectorized(wd_mean, wdr_mean, threshold):
    """
    Vectorized calculation of ratios and differences with threshold logic.
    """
    nlat, nlon = wd_mean.shape
    ratios = np.full((nlat, nlon), np.nan)
    differences = np.full((nlat, nlon), np.nan)
    
    for i in prange(nlat):
        for j in range(nlon):
            wd_val = wd_mean[i, j]
            wdr_val = wdr_mean[i, j]
            
            if (not np.isnan(wd_val) and not np.isnan(wdr_val) and 
                wd_val > threshold):
                
                # Ratio: WD/WDXXR (efficiency metric) - should be ≤ 1.0
                if wdr_val > 0:
                    ratio_val = wd_val / wdr_val
                    # Ensure ratio doesn't exceed 1.0 (WD should ≤ WDXXR)
                    ratios[i, j] = min(ratio_val, 1.0)
                
                # Difference: WDXXR - WD (temporal dispersion) - should be ≥ 0
                diff_val = wdr_val - wd_val
                # Ensure difference is non-negative (WDXXR should ≥ WDXX)
                differences[i, j] = max(diff_val, 0.0)
    
    return ratios, differences

def create_optimized_land_ocean_mask(ds, mask_type='both', validate=True):
    """Optimized land/ocean mask creation."""
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
    
    land_regions = [
        ((lat_grid >= 25) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -50)),
        ((lat_grid >= -60) & (lat_grid <= 15) & (lon_grid >= -85) & (lon_grid <= -30)),
        ((lat_grid >= 35) & (lat_grid <= 75) & (lon_grid >= -15) & (lon_grid <= 45)),
        ((lat_grid >= 5) & (lat_grid <= 75) & (lon_grid >= 25) & (lon_grid <= 180)),
        ((lat_grid >= -35) & (lat_grid <= 40) & (lon_grid >= -20) & (lon_grid <= 55)),
        ((lat_grid >= -50) & (lat_grid <= -5) & (lon_grid >= 110) & (lon_grid <= 180)),
        ((lat_grid >= 60) & (lat_grid <= 85) & (lon_grid >= -75) & (lon_grid <= -10)),
        (lat_grid <= -60),
    ]
    
    for region in land_regions:
        land_mask |= region
    
    if validate:
        land_pct = np.sum(land_mask) / land_mask.size * 100
        print(f"  Fallback land percentage: {land_pct:.1f}%")
    
    return land_mask if mask_type == 'land' else ~land_mask

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
                            year_data[var] = ds[var].values
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
    """Load WDXXR chronological data."""
    wd50r_dir = Path(wd50r_dir)
    
    print(f"Loading WDXXR data from: {wd50r_dir.absolute()}")
    print(f"Target percentiles: {percentiles}")
    
    if years is None:
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
                            
                            wdxxr_data_list.append(wd_values.values)
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

def create_multi_threshold_comparison_figure(comparison_data, lat_coords, lon_coords, output_dir, 
                                           mask_type='both', threshold=0.0, percentiles=[25, 50, 75]):
    """
    Create multi-threshold WDXX vs WDXXR comparison figure.
    """
    # Create mask
    dummy_ds = xr.Dataset(coords={'latitude': lat_coords, 'longitude': lon_coords})
    mask = create_optimized_land_ocean_mask(dummy_ds, mask_type)
    
    # Pre-calculate local ranges for each subplot
    print("Calculating local color ranges for each subplot...")

    # Calculate local ranges for each percentile
    local_ranges = {}
    for percentile in percentiles:
        ratio_key = f'ratio_{percentile}'
        diff_key = f'difference_{percentile}'

        # Local ratio range
        if ratio_key in comparison_data:
            ratio_masked = comparison_data[ratio_key].copy()
            ratio_masked[~mask] = np.nan
            valid_ratios = ratio_masked[~np.isnan(ratio_masked)]
            if len(valid_ratios) > 0:
                ratio_vmin = np.percentile(valid_ratios, 1)  # 1st percentile
                ratio_vmax = min(1.0, np.percentile(valid_ratios, 99))  # 99th percentile, capped at 1.0
                ratio_vmin = max(0.1, ratio_vmin)
                ratio_vmax = max(0.8, ratio_vmax)
            else:
                ratio_vmin, ratio_vmax = 0.1, 1.0
        else:
            ratio_vmin, ratio_vmax = 0.1, 1.0

        # Local difference range
        if diff_key in comparison_data:
            diff_masked = comparison_data[diff_key].copy()
            diff_masked[~mask] = np.nan
            valid_diffs = diff_masked[~np.isnan(diff_masked)]
            if len(valid_diffs) > 0:
                diff_vmin = np.percentile(valid_diffs, 1)  # 1st percentile
                diff_vmax = np.percentile(valid_diffs, 99)  # 99th percentile
                diff_vmin = max(0.0, diff_vmin)  # Ensure non-negative
                diff_vmax = max(5.0, diff_vmax)  # Ensure minimum range
            else:
                diff_vmin, diff_vmax = 0.0, 50.0
        else:
            diff_vmin, diff_vmax = 0.0, 50.0

        local_ranges[percentile] = {
            'ratio_vmin': ratio_vmin,
            'ratio_vmax': ratio_vmax,
            'diff_vmin': diff_vmin,
            'diff_vmax': diff_vmax
        }

        print(f"{percentile}% - Ratio: [{ratio_vmin:.3f}, {ratio_vmax:.3f}], Difference: [{diff_vmin:.1f}, {diff_vmax:.1f}] days")
    
    # Create figure
    fig = plt.figure(figsize=(12, 15))  # Taller for 3 rows
    proj = ccrs.Robinson(central_longitude=0)
    
    for row_idx, percentile in enumerate(percentiles):
        # Get data for this percentile
        ratio_key = f'ratio_{percentile}'
        diff_key = f'difference_{percentile}'

        if ratio_key not in comparison_data or diff_key not in comparison_data:
            print(f"Warning: Data for percentile {percentile} not found")
            continue

        # Get local ranges for this percentile
        ranges = local_ranges[percentile]
        ratio_vmin, ratio_vmax = ranges['ratio_vmin'], ranges['ratio_vmax']
        diff_vmin, diff_vmax = ranges['diff_vmin'], ranges['diff_vmax']

        ratio_data = comparison_data[ratio_key]
        diff_data = comparison_data[diff_key]

        # Ratio plot (left column)
        ax_ratio = plt.subplot(3, 2, row_idx * 2 + 1, projection=proj)

        # Map features
        ax_ratio.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
        ax_ratio.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax_ratio.set_global()

        # Apply mask to ratio data
        ratio_masked = ratio_data.copy()
        ratio_masked[~mask] = np.nan

        # Calculate actual data range for ratio
        valid_ratio = ratio_masked[~np.isnan(ratio_masked)]
        ratio_actual_min = np.min(valid_ratio) if len(valid_ratio) > 0 else 0
        ratio_actual_max = np.max(valid_ratio) if len(valid_ratio) > 0 else 1
        
        # Use local ratio range
        ratio_levels = np.linspace(ratio_vmin, ratio_vmax, 20)
        im_ratio = ax_ratio.contourf(lon_coords, lat_coords, ratio_masked,
                                   levels=ratio_levels, cmap='viridis',
                                   transform=ccrs.PlateCarree(), extend='both')

        # Create individual colorbar for ratio with compact positioning
        # Get subplot position and place colorbar just below it
        ax_ratio_pos = ax_ratio.get_position()

        # Compact colorbar dimensions (following median_change.py example)
        cbar_height = 0.015  # Slightly taller for better visibility
        cbar_gap = -0.008    # Reduced gap to bring colorbar closer to subplot

        ratio_cbar_ax = fig.add_axes([0.08, ax_ratio_pos.y0 - cbar_gap - cbar_height,
                                     0.35, cbar_height])
        
        # Create normalized colorbar with extend='both'
        norm_ratio = mcolors.Normalize(vmin=ratio_vmin, vmax=ratio_vmax)
        dummy_ratio = plt.cm.ScalarMappable(norm=norm_ratio, cmap='viridis')
        ratio_cbar = plt.colorbar(dummy_ratio, cax=ratio_cbar_ax, orientation='horizontal', extend='both')
        
        # Comprehensive colorbar label
        ratio_cbar_label = f'WD{percentile}/WD{percentile}R\nColor: 1–99 percentile | Actual: {ratio_actual_min:.2f}–{ratio_actual_max:.2f}'
        ratio_cbar.set_label(ratio_cbar_label, fontsize=7, labelpad=1, linespacing=1.2)
        
        # Set ticks strictly within range
        ratio_ticks = np.linspace(ratio_vmin, ratio_vmax, 5)
        ratio_cbar.set_ticks(ratio_ticks)
        ratio_cbar.set_ticklabels([f'{tick:.2f}' for tick in ratio_ticks])
        ratio_cbar.ax.tick_params(labelsize=7)

        # Title for ratio
        ax_ratio.set_title(f'({chr(97 + row_idx * 2)}) WD{percentile}/WD{percentile}R',
                          fontsize=13, fontweight='bold', pad=10)

        # Difference plot (right column)
        ax_diff = plt.subplot(3, 2, row_idx * 2 + 2, projection=proj)

        # Map features
        ax_diff.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
        ax_diff.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
        ax_diff.set_global()

        # Apply mask to difference data
        diff_masked = diff_data.copy()
        diff_masked[~mask] = np.nan

        # Calculate actual data range for difference
        valid_diff = diff_masked[~np.isnan(diff_masked)]
        diff_actual_min = np.min(valid_diff) if len(valid_diff) > 0 else 0
        diff_actual_max = np.max(valid_diff) if len(valid_diff) > 0 else 1
        
        # Use local difference range
        diff_levels = np.linspace(diff_vmin, diff_vmax, 20)
        im_diff = ax_diff.contourf(lon_coords, lat_coords, diff_masked,
                                 levels=diff_levels, cmap='plasma',
                                 transform=ccrs.PlateCarree(), extend='both')

        # Create individual colorbar for difference with compact positioning
        # Get subplot position and place colorbar just below it
        ax_diff_pos = ax_diff.get_position()

        # Use same compact dimensions as ratio colorbar
        cbar_height = 0.015
        cbar_gap = -0.008

        diff_cbar_ax = fig.add_axes([0.58, ax_diff_pos.y0 - cbar_gap - cbar_height,
                                    0.35, cbar_height])
        
        # Create normalized colorbar with extend='both'
        norm_diff = mcolors.Normalize(vmin=diff_vmin, vmax=diff_vmax)
        dummy_diff = plt.cm.ScalarMappable(norm=norm_diff, cmap='plasma')
        diff_cbar = plt.colorbar(dummy_diff, cax=diff_cbar_ax, orientation='horizontal', extend='both')
        
        # Comprehensive colorbar label
        diff_cbar_label = f'WD{percentile}R - WD{percentile} (days)\nColor: 1–99 percentile | Actual: {diff_actual_min:.1f}–{diff_actual_max:.1f} days'
        diff_cbar.set_label(diff_cbar_label, fontsize=7, labelpad=1, linespacing=1.2)
        
        # Set ticks strictly within range
        diff_ticks = np.linspace(diff_vmin, diff_vmax, 5)
        diff_ticks = np.round(diff_ticks).astype(int)
        diff_cbar.set_ticks(diff_ticks)
        diff_cbar.set_ticklabels([f'{tick}' for tick in diff_ticks])
        diff_cbar.ax.tick_params(labelsize=7)

        # Title for difference
        ax_diff.set_title(f'({chr(97 + row_idx * 2 + 1)}) WD{percentile}R - WD{percentile}',
                         fontsize=13, fontweight='bold', pad=10)

        # Add statistics text for both plots
        valid_ratio = ratio_masked[~np.isnan(ratio_masked)]
        valid_diff = diff_masked[~np.isnan(diff_masked)]

        '''
        if len(valid_ratio) > 0:
            ratio_mean = np.mean(valid_ratio)
            ratio_std = np.std(valid_ratio)
            ax_ratio.text(0.02, 0.98, f'Mean: {ratio_mean:.2f} ± {ratio_std:.2f}',
                         transform=ax_ratio.transAxes,
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
                         verticalalignment='top', fontsize=10)

        if len(valid_diff) > 0:
            diff_mean = np.mean(valid_diff)
            diff_std = np.std(valid_diff)
            ax_diff.text(0.02, 0.98, f'Mean: {diff_mean:.1f} ± {diff_std:.1f} days',
                        transform=ax_diff.transAxes,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
                        verticalalignment='top', fontsize=10)
        '''

    # Main title
    threshold_text = f"(Analysis where mean intensity index > {threshold} days)" if threshold > 0 else ""
    title_text = f'Multi-Threshold Intensity vs Chronological Precipitation Comparison\n{threshold_text}'
    
    fig.suptitle(title_text, fontsize=18, fontweight='bold', y=0.96)
    
    # Compact layout following median_change.py example
    plt.tight_layout()
    # Adjust bottom margin to accommodate colorbars (cbar_height + cbar_gap + some padding)
    bottom_margin = 0.18  # Following median_change.py for consistency
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=bottom_margin,
                       hspace=0.35, wspace=0.05)
    
    # Save
    threshold_str = f"_thresh{threshold:.0f}" if threshold > 0 else ""
    filename = f'multi_threshold_intensity_chronological_comparison_{mask_type}{threshold_str}.png'
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    return output_path

def main():
    """Main function for multi-threshold comparison analysis."""
    parser = argparse.ArgumentParser(description='Multi-threshold WD vs WDXXR comparison analysis')
    
    # Data directories
    parser.add_argument('--enhanced-dir', default='data/processed/enhanced_concentration_indices',
                       help='Directory containing enhanced index files (for WDXX)')
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices', 
                       help='Directory containing WDXXR index files')
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures and data')
    
    # Analysis parameters
    parser.add_argument('--percentiles', nargs='+', type=int, default=[25, 50, 75],
                       help='Percentiles to analyze (default: 25 50 75)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='both',
                       help='Analysis domain (default: both)')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Minimum WD value to calculate ratio/difference (default: 0.0)')
    
    # Processing parameters  
    parser.add_argument('--n-processes', type=int, default=None,
                       help='Number of processes for parallel computation (default: auto)')
    
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
    print("MULTI-THRESHOLD INTENSITY VS CHRONOLOGICAL COMPARISON")
    print("="*80)
    print(f"Enhanced directory: {args.enhanced_dir}")
    print(f"WDXXR directory: {args.wd50r_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Percentiles: {args.percentiles}")
    print(f"Analysis domain: {args.mask_type}")
    print(f"Threshold: {args.threshold} days")
    print("="*80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load enhanced data (WDXX)
        print("Loading enhanced concentration data (WDXX)...")
        wd_vars = [f'WD{p}' for p in args.percentiles]
        enhanced_ds = load_enhanced_data_optimized(args.enhanced_dir, wd_vars)
        
        print("Loading chronological data (WDXXR)...")
        wd50r_ds = load_wd50r_data_optimized(args.wd50r_dir, args.percentiles)
        
        # Find common years and align datasets
        common_years = sorted(set(enhanced_ds.year.values) & set(wd50r_ds.year.values))
        if not common_years:
            raise ValueError("No common years found between WDXX and WDXXR datasets")
        
        print(f"Common years: {len(common_years)} years ({common_years[0]}-{common_years[-1]})")
        
        # Align datasets
        enhanced_aligned = enhanced_ds.sel(year=common_years)
        wd50r_aligned = wd50r_ds.sel(year=common_years)
        
        # Ensure same spatial coordinates
        if not np.array_equal(enhanced_aligned.latitude.values, wd50r_aligned.latitude.values):
            raise ValueError("WDXX and WDXXR datasets have different latitude coordinates")
        if not np.array_equal(enhanced_aligned.longitude.values, wd50r_aligned.longitude.values):
            raise ValueError("WDXX and WDXXR datasets have different longitude coordinates")
        
        lat_coords = enhanced_aligned.latitude.values
        lon_coords = enhanced_aligned.longitude.values
        
        # Calculate multi-year means for all percentiles
        print("Calculating multi-year means...")
        comparison_data = {}
        
        for percentile in args.percentiles:
            wd_var = f'WD{percentile}'
            wdr_var = f'WD{percentile}R'
            
            print(f"Processing {percentile}% threshold...")
            
            # Calculate means across years
            wd_mean = enhanced_aligned[wd_var].mean(dim='year').values
            wdr_mean = wd50r_aligned[wdr_var].mean(dim='year').values
            
            # Calculate ratios and differences
            ratios, differences = calculate_ratios_differences_vectorized(
                wd_mean, wdr_mean, args.threshold
            )
            
            comparison_data[f'ratio_{percentile}'] = ratios
            comparison_data[f'difference_{percentile}'] = differences
            
            # Statistics
            valid_ratios = ratios[~np.isnan(ratios)]
            valid_diffs = differences[~np.isnan(differences)]
            
            if len(valid_ratios) > 0:
                ratio_min, ratio_max = np.min(valid_ratios), np.max(valid_ratios)
                print(f"  {percentile}% - Ratios: {len(valid_ratios)} pixels, "
                      f"mean={np.mean(valid_ratios):.3f}, std={np.std(valid_ratios):.3f}, "
                      f"range=[{ratio_min:.3f}, {ratio_max:.3f}]")
            if len(valid_diffs) > 0:
                diff_min, diff_max = np.min(valid_diffs), np.max(valid_diffs)
                print(f"  {percentile}% - Differences: {len(valid_diffs)} pixels, "
                      f"mean={np.mean(valid_diffs):.1f} days, std={np.std(valid_diffs):.1f} days, "
                      f"range=[{diff_min:.1f}, {diff_max:.1f}] days")
        
        # Create figure
        print("Creating multi-threshold comparison figure...")
        output_path = create_multi_threshold_comparison_figure(
            comparison_data, lat_coords, lon_coords, output_dir,
            args.mask_type, args.threshold, args.percentiles
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Figure saved: {output_path}")
        print("Analysis components:")
        for percentile in args.percentiles:
            print(f"- {percentile}%: Efficiency ratio (WD{percentile}/WD{percentile}R) and temporal dispersion (WD{percentile}R-WD{percentile})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())