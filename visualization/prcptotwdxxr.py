#!/usr/bin/env python3
"""
Create figure comparing precipitation intensity during concentration periods:
(a) I50 = (PRCPTOT * 0.5) / WD50 - Intensity during wettest days
(b) I50R = (PRCPTOT * 0.5) / WD50R - Intensity during consecutive clustering

This script loads ETCCDI, enhanced concentration indices, and WD50R data
to calculate and visualize these derived intensity metrics.
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
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

def create_land_ocean_mask(ds):
    """Create land/ocean mask using regionmask or simple heuristic."""
    if HAS_REGIONMASK:
        try:
            land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(
                ds.longitude, ds.latitude
            )
            return ~np.isnan(land_mask)
        except Exception as e:
            print(f"Regionmask failed: {e}. Using simple heuristic.")
    
    # Simple heuristic: assume areas with reasonable precipitation are land
    # This is a fallback and may not be perfect
    if 'PRCPTOT' in ds.data_vars:
        precip_data = ds.PRCPTOT.values
        # Land typically has some precipitation, oceans in grid often show as 0 or very low
        land_mask = (precip_data > 10) & (precip_data < 5000)  # mm/year reasonable range
    else:
        print("Warning: No precipitation data for land mask. Using all points.")
        land_mask = np.ones(ds.longitude.shape + ds.latitude.shape, dtype=bool)
    
    return land_mask

def load_multi_year_data(data_dirs, years):
    """Load and combine multi-year datasets."""
    print(f"Loading data for years: {years[0]}-{years[-1]}")
    
    # Initialize data containers
    etccdi_data = []
    enhanced_data = []
    wd50r_data = []
    
    # Load data for each year
    missing_years = {'etccdi': [], 'enhanced': [], 'wd50r': []}
    
    for year in years:
        try:
            # ETCCDI data (correct filename pattern)
            etccdi_file = Path(data_dirs['etccdi']) / f'etccdi_precipitation_indices_{year}.nc'
            if etccdi_file.exists():
                etccdi_ds = xr.open_dataset(etccdi_file)
                # Handle timedelta variables if they exist
                for var in ['WD50', 'wet_days', 'very_wet_days']:
                    if var in etccdi_ds.data_vars and 'timedelta' in str(etccdi_ds[var].dtype):
                        etccdi_ds[var] = etccdi_ds[var] / np.timedelta64(1, 'D')
                etccdi_data.append(etccdi_ds)
            else:
                missing_years['etccdi'].append(year)
            
            # Enhanced concentration data
            enhanced_file = Path(data_dirs['enhanced']) / f'enhanced_concentration_indices_{year}_all.nc'
            if enhanced_file.exists():
                enhanced_ds = xr.open_dataset(enhanced_file)
                # Handle timedelta variables if they exist in enhanced data
                for var in ['WD10', 'WD25', 'WD50', 'WD75', 'WD90']:
                    if var in enhanced_ds.data_vars and 'timedelta' in str(enhanced_ds[var].dtype):
                        enhanced_ds[var] = enhanced_ds[var] / np.timedelta64(1, 'D')
                enhanced_data.append(enhanced_ds)
            else:
                missing_years['enhanced'].append(year)
            
            # WD50R data (look for different possible filename patterns)
            wd50r_patterns = [
                f'wd50r_indices_{year}_P25_50_75.nc',
                f'wd50r_indices_{year}_P50.nc',
                f'wd50r_indices_{year}.nc'
            ]
            
            wd50r_found = False
            for pattern in wd50r_patterns:
                wd50r_file = Path(data_dirs['wd50r']) / pattern
                if wd50r_file.exists():
                    wd50r_ds = xr.open_dataset(wd50r_file)
                    # Handle timedelta variables if they exist in wd50r data
                    for var in ['WD25R', 'WD50R', 'WD75R', 'WD90R']:
                        if var in wd50r_ds.data_vars and 'timedelta' in str(wd50r_ds[var].dtype):
                            wd50r_ds[var] = wd50r_ds[var] / np.timedelta64(1, 'D')
                    wd50r_data.append(wd50r_ds)
                    wd50r_found = True
                    break
            
            if not wd50r_found:
                missing_years['wd50r'].append(year)
                    
        except Exception as e:
            print(f"Error loading data for year {year}: {e}")
            continue
    
    # Report missing data
    for dataset_type, missing in missing_years.items():
        if missing:
            print(f"Warning: Missing {dataset_type} data for {len(missing)} years: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Combine datasets
    combined_data = {}
    
    if etccdi_data:
        print(f"Combining {len(etccdi_data)} ETCCDI files...")
        etccdi_combined = xr.concat(etccdi_data, dim='year')
        combined_data['etccdi'] = etccdi_combined.mean(dim='year')  # Multi-year mean
        for ds in etccdi_data:
            ds.close()
    
    if enhanced_data:
        print(f"Combining {len(enhanced_data)} Enhanced files...")
        enhanced_combined = xr.concat(enhanced_data, dim='year')
        combined_data['enhanced'] = enhanced_combined.mean(dim='year')
        for ds in enhanced_data:
            ds.close()
    
    if wd50r_data:
        print(f"Combining {len(wd50r_data)} WD50R files...")
        wd50r_combined = xr.concat(wd50r_data, dim='year')
        combined_data['wd50r'] = wd50r_combined.mean(dim='year')
        for ds in wd50r_data:
            ds.close()
    
    return combined_data

def calculate_intensity_metrics(combined_data):
    """Calculate I50 and I50R intensity metrics."""
    print("Calculating intensity metrics...")
    
    # Get required variables - try multiple sources for PRCPTOT
    prcptot = None
    wd50 = None
    wd50r = None
    
    try:
        # Try to get PRCPTOT from ETCCDI data first
        if 'etccdi' in combined_data and 'PRCPTOT' in combined_data['etccdi'].data_vars:
            prcptot = combined_data['etccdi']['PRCPTOT'].values
            print("Using PRCPTOT from ETCCDI data")
        # Fallback to enhanced data
        elif 'enhanced' in combined_data and 'PRCPTOT' in combined_data['enhanced'].data_vars:
            prcptot = combined_data['enhanced']['PRCPTOT'].values
            print("Using PRCPTOT from enhanced data")
        else:
            raise KeyError("PRCPTOT not found in any dataset")
        
        # WD50 from enhanced data
        if 'enhanced' in combined_data and 'WD50' in combined_data['enhanced'].data_vars:
            wd50 = combined_data['enhanced']['WD50'].values
            print("Using WD50 from enhanced data")
        else:
            print("Warning: WD50 not found in enhanced data. Using placeholder.")
            wd50 = np.full_like(prcptot, np.nan)
        
        # WD50R from wd50r data
        if 'wd50r' in combined_data and 'WD50R' in combined_data['wd50r'].data_vars:
            wd50r = combined_data['wd50r']['WD50R'].values
            print("Using WD50R from wd50r data")
        else:
            raise KeyError("WD50R not found in wd50r data - this is required!")
            
    except KeyError as e:
        print(f"Missing required variable: {e}")
        print(f"Available datasets: {list(combined_data.keys())}")
        if combined_data:
            for dataset_name, dataset in combined_data.items():
                print(f"  {dataset_name} variables: {list(dataset.data_vars)}")
        return None, None
    
    # Calculate intensity metrics
    # I50 = Intensity during wettest days (mm/day)
    i50 = np.where((prcptot > 0) & (wd50 > 0), (prcptot * 0.5) / wd50, np.nan)
    
    # I50R = Intensity during consecutive clustering (mm/day)  
    i50r = np.where((prcptot > 0) & (wd50r > 0), (prcptot * 0.5) / wd50r, np.nan)
    
    print(f"I50 range: {np.nanmin(i50):.1f} - {np.nanmax(i50):.1f} mm/day")
    print(f"I50R range: {np.nanmin(i50r):.1f} - {np.nanmax(i50r):.1f} mm/day")
    
    return i50, i50r

def create_intensity_figure(combined_data, i50, i50r, output_dir, mask_type='land'):
    """Create the main intensity comparison figure."""
    print("Creating intensity comparison figure...")
    
    # Get coordinates from the first available dataset
    lats = lons = None
    for dataset_name in ['etccdi', 'enhanced', 'wd50r']:
        if dataset_name in combined_data:
            lats = combined_data[dataset_name]['latitude'].values
            lons = combined_data[dataset_name]['longitude'].values
            print(f"Using coordinates from {dataset_name} dataset")
            break
    
    if lats is None or lons is None:
        raise ValueError("No coordinate data found in any dataset!")
    
    # Create land mask using the first available dataset
    first_dataset = list(combined_data.values())[0]
    land_mask = create_land_ocean_mask(first_dataset)
    
    # Apply mask
    if mask_type == 'land':
        mask = land_mask
        title_suffix = 'Land Areas'
    elif mask_type == 'ocean':
        mask = ~land_mask  
        title_suffix = 'Ocean Areas'
    else:
        mask = np.ones_like(land_mask, dtype=bool)
        title_suffix = 'Global'
    
    # Mask data
    i50_masked = np.where(mask, i50, np.nan)
    i50r_masked = np.where(mask, i50r, np.nan)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Color scheme - use same colormap for both panels for comparison
    # Calculate actual data range
    all_intensity_data = np.concatenate([i50_masked.flatten(), i50r_masked.flatten()])
    valid_intensity = all_intensity_data[~np.isnan(all_intensity_data)]
    actual_min = np.min(valid_intensity) if len(valid_intensity) > 0 else 0
    actual_max = np.max(valid_intensity) if len(valid_intensity) > 0 else 1
    
    vmin = 0
    vmax = np.nanpercentile(all_intensity_data, 95)
    percentile_range = "0–95"
    cmap = plt.cm.viridis
    levels = np.linspace(vmin, vmax, 20)
    
    # Panel (a): I50 - Intensity during wettest days
    ax1 = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
    ax1.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    
    im1 = ax1.contourf(lons, lats, i50_masked, levels=levels, cmap=cmap, 
                       transform=ccrs.PlateCarree(), extend='max')
    
    ax1.set_title('(a) I50: Precipitation Intensity During Wettest Days\n' +
                  f'I50 = (PRCPTOT × 0.5) / WD50 - {title_suffix}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Panel (b): I50R - Intensity during consecutive clustering  
    ax2 = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax2.set_global()
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray')
    ax2.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray')
    
    im2 = ax2.contourf(lons, lats, i50r_masked, levels=levels, cmap=cmap,
                       transform=ccrs.PlateCarree(), extend='max')
    
    ax2.set_title('(b) I50R: Precipitation Intensity During Consecutive Clustering\n' +
                  f'I50R = (PRCPTOT × 0.5) / WD50R - {title_suffix}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    
    # Create normalized colorbar with extend='both' for extremes
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    dummy_im = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(dummy_im, cax=cbar_ax, extend='both')
    
    # Comprehensive colorbar label
    cbar_label = f'mm/day\nColor: {percentile_range} percentile\nActual: {actual_min:.1f} to {actual_max:.1f}'
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold', linespacing=1.2)
    
    # Set ticks strictly within [vmin, vmax]
    tick_locs = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f'{val:.1f}' for val in tick_locs])
    cbar.ax.tick_params(labelsize=11)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08, hspace=0.2)
    
    # Add statistics text boxes
    # Calculate statistics for valid data
    i50_valid = i50_masked[~np.isnan(i50_masked)]
    i50r_valid = i50r_masked[~np.isnan(i50r_masked)]
    
    if len(i50_valid) > 0:
        i50_stats = f'Mean: {np.mean(i50_valid):.1f}\nMedian: {np.median(i50_valid):.1f}\nMax: {np.max(i50_valid):.1f}'
        ax1.text(0.02, 0.02, i50_stats, transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    if len(i50r_valid) > 0:
        i50r_stats = f'Mean: {np.mean(i50r_valid):.1f}\nMedian: {np.median(i50r_valid):.1f}\nMax: {np.max(i50r_valid):.1f}'
        ax2.text(0.02, 0.02, i50r_stats, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Save figure
    filename = f'intensity_concentration_comparison_{mask_type}.png'
    output_path = Path(output_dir) / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved: {output_path}")
    return output_path

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create intensity during concentration figure',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data directories
    parser.add_argument('--etccdi-dir', default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI indices')
    parser.add_argument('--enhanced-dir', default='data/processed/enhanced_concentration_indices', 
                       help='Directory containing enhanced concentration indices')
    parser.add_argument('--wd50r-dir', default='data/processed/wd50r_indices',
                       help='Directory containing WD50R indices')
    
    # Analysis parameters
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year for analysis (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--mask-type', choices=['land', 'ocean', 'both'], default='land',
                       help='Analysis domain (default: land)')
    
    # Output
    parser.add_argument('--output-dir', default='figures/',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PRECIPITATION INTENSITY DURING CONCENTRATION ANALYSIS")
    print("="*80)
    print(f"Analysis period: {args.start_year}-{args.end_year}")
    print(f"Mask type: {args.mask_type}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up data directories
    data_dirs = {
        'etccdi': args.etccdi_dir,
        'enhanced': args.enhanced_dir, 
        'wd50r': args.wd50r_dir
    }
    
    # Check directories exist and show sample files
    for name, directory in data_dirs.items():
        dir_path = Path(directory)
        if dir_path.exists():
            nc_files = list(dir_path.glob('*.nc'))
            print(f"{name.upper()} directory: {len(nc_files)} .nc files found")
            if nc_files:
                print(f"  Sample files: {[f.name for f in nc_files[:3]]}")
        else:
            print(f"WARNING: {name.upper()} directory does not exist: {directory}")
    
    # Define years
    years = list(range(args.start_year, args.end_year + 1))
    
    # Load data
    combined_data = load_multi_year_data(data_dirs, years)
    
    if not combined_data:
        print("Error: No data loaded successfully!")
        return 1
        
    # Calculate intensity metrics
    i50, i50r = calculate_intensity_metrics(combined_data)
    
    if i50 is None or i50r is None:
        print("Error: Could not calculate intensity metrics!")
        return 1
    
    # Create figure
    output_path = create_intensity_figure(combined_data, i50, i50r, output_dir, args.mask_type)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Figure saved: {output_path}")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())