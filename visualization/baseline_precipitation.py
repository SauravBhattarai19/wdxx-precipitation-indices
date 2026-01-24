#!/usr/bin/env python3
"""
Enhanced Precipitation Analysis Figure with Side-by-Side Layout
- Each row: Map + Latitudinal variation (sharing latitude axis)
- Latitudinal plots show mean ± standard deviation across longitudes
- Row 1: 90th percentile precipitation
- Row 2: Annual mean precipitation  
- Row 3: Annual max precipitation

Features:
- Shared latitude axes between map and line plots
- Mean ± std dev visualization in latitudinal plots
- Advanced land/ocean masking
- Academic-quality BrBG color scheme
- Pointy colorbars on both ends
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
import argparse
from scipy import stats

# Try to import advanced masking libraries
try:
    import regionmask
    HAS_REGIONMASK = True
except ImportError:
    HAS_REGIONMASK = False
    print("Note: regionmask not available - using cartopy fallback")

warnings.filterwarnings('ignore')

def create_land_ocean_mask(ds, mask_type='both'):
    """
    Create accurate land/ocean mask using regionmask (if available) or cartopy.
    
    Args:
        ds: xarray Dataset with latitude/longitude coordinates
        mask_type: 'land', 'ocean', or 'both'
    
    Returns:
        mask: Boolean array where True = include pixel, False = exclude
    """
    if mask_type == 'both':
        return np.ones((len(ds.latitude), len(ds.longitude)), dtype=bool)
    
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    print(f"Creating {mask_type} mask...")
    print(f"  Grid: {len(lat)} x {len(lon)} pixels")
    
    if HAS_REGIONMASK:
        try:
            # Use regionmask with Natural Earth land boundaries (much faster!)
            print("  Using regionmask with Natural Earth boundaries...")
            
            # Create a dummy dataset with coordinates named as regionmask expects
            dummy_ds = xr.Dataset(coords={'lat': lat, 'lon': lon})
            
            # Get Natural Earth land boundaries
            land_mask_rm = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(dummy_ds)
            
            # Convert to boolean (regionmask returns integers)
            land_mask = ~np.isnan(land_mask_rm.values)
            
            print(f"  Land pixels: {np.sum(land_mask):,} ({np.sum(land_mask)/land_mask.size*100:.1f}%)")
            print(f"  Ocean pixels: {np.sum(~land_mask):,} ({np.sum(~land_mask)/land_mask.size*100:.1f}%)")
            
            if mask_type == 'land':
                return land_mask
            elif mask_type == 'ocean':
                return ~land_mask
            else:
                return np.ones_like(land_mask, dtype=bool)
                
        except Exception as e:
            print(f"  regionmask failed: {e}")
            print("  Falling back to cartopy...")
    
    # Fallback: Use cartopy's built-in land feature
    print("  Using cartopy Natural Earth features...")
    
    try:
        # Use cartopy's natural earth land feature
        import cartopy.io.shapereader as shpreader
        
        # Get land geometries
        land_shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
        land_geoms = list(shpreader.Reader(land_shp).geometries())
        
        # Create mask efficiently
        land_mask = np.zeros((len(lat), len(lon)), dtype=bool)
        
        # Create coordinate meshgrid
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Vectorized point-in-polygon test (more efficient)
        from shapely.geometry import Point
        from shapely.prepared import prep
        
        # Prepare geometries for faster intersection
        prepared_geoms = [prep(geom) for geom in land_geoms]
        
        # Check each point
        for i in range(len(lat)):
            for j in range(len(lon)):
                point = Point(lon[j], lat[i])
                for prep_geom in prepared_geoms:
                    if prep_geom.contains(point):
                        land_mask[i, j] = True
                        break
        
        print(f"  Land pixels: {np.sum(land_mask):,} ({np.sum(land_mask)/land_mask.size*100:.1f}%)")
        print(f"  Ocean pixels: {np.sum(~land_mask):,} ({np.sum(~land_mask)/land_mask.size*100:.1f}%)")
        
        if mask_type == 'land':
            return land_mask
        elif mask_type == 'ocean':
            return ~land_mask
        else:
            return np.ones_like(land_mask, dtype=bool)
            
    except Exception as e:
        print(f"  Error creating land mask: {e}")
        print("  Using simplified approximation...")
        
        # Last resort: very simplified land/ocean approximation
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        
        # Very basic land approximation (major continents only)
        land_mask = np.zeros_like(lat_grid, dtype=bool)
        
        # Continents (very rough)
        land_mask |= (lat_grid >= -60) & (lat_grid <= 75) & (lon_grid >= -170) & (lon_grid <= -30)  # Americas
        land_mask |= (lat_grid >= -35) & (lat_grid <= 75) & (lon_grid >= -20) & (lon_grid <= 180)   # Eurasia+Africa
        land_mask |= (lat_grid >= -50) & (lat_grid <= -10) & (lon_grid >= 110) & (lon_grid <= 180)  # Australia
        
        # Remove obvious ocean areas
        land_mask &= ~((lat_grid >= -10) & (lat_grid <= 10) & (lon_grid >= -40) & (lon_grid <= 20))  # Atlantic
        land_mask &= ~((lat_grid >= -10) & (lat_grid <= 10) & (lon_grid >= 80) & (lon_grid <= 120))   # Indian Ocean
        
        if mask_type == 'land':
            return land_mask
        elif mask_type == 'ocean':
            return ~land_mask
        else:
            return np.ones_like(land_mask, dtype=bool)

def apply_mask_and_adjust_colorbar(data, mask, percentile_range=(0.5, 99.5)):
    """
    Apply mask to data and calculate appropriate colorbar range.
    
    Args:
        data: numpy array of data values
        mask: boolean mask (True = include)
        percentile_range: tuple of (low, high) percentiles for colorbar
    
    Returns:
        masked_data: data with mask applied (NaN where mask is False)
        vmin, vmax: colorbar range values
        valid_count: number of valid pixels
    """
    # Apply mask
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    
    # Calculate colorbar range from masked data
    valid_data = masked_data[~np.isnan(masked_data)]
    
    if len(valid_data) == 0:
        return masked_data, 0, 1, 0
    
    vmin = np.percentile(valid_data, percentile_range[0])
    vmax = np.percentile(valid_data, percentile_range[1])
    
    return masked_data, vmin, vmax, len(valid_data)

def load_precipitation_percentiles(percentile_file):
    """Load precipitation percentiles data."""
    if not Path(percentile_file).exists():
        raise FileNotFoundError(f"Percentile file not found: {percentile_file}")
    
    ds = xr.open_dataset(percentile_file)
    
    if 'precip_p90' not in ds.data_vars:
        raise ValueError("P90 precipitation percentiles not found in file")
    
    # Take annual mean of day-of-year climatology
    p90_annual = ds.precip_p90.mean(dim='dayofyear')
    
    return p90_annual, ds.latitude.values, ds.longitude.values

def load_multi_year_precipitation_stats(etccdi_dir, start_year=1980, end_year=2024):
    """Load and calculate multi-year precipitation statistics."""
    etccdi_dir = Path(etccdi_dir)
    
    annual_mean_list = []
    annual_max_list = []
    years_loaded = []
    
    for year in range(start_year, end_year + 1):
        file_path = etccdi_dir / f'etccdi_precipitation_indices_{year}.nc'
        
        if not file_path.exists():
            continue
            
        try:
            ds = xr.open_dataset(file_path)
            
            if 'PRCPTOT' in ds.data_vars:
                annual_mean_list.append(ds.PRCPTOT.values)
                years_loaded.append(year)
            
            if 'max_daily_precip' in ds.data_vars:
                annual_max_list.append(ds.max_daily_precip.values)
            
            # Get coordinates from first file
            if len(years_loaded) == 1:
                lat = ds.latitude.values
                lon = ds.longitude.values
            
            ds.close()
            
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if not annual_mean_list:
        raise ValueError(f"No ETCCDI files found in {etccdi_dir} for years {start_year}-{end_year}")
    
    print(f"Loaded {len(years_loaded)} years: {years_loaded[0]}-{years_loaded[-1]}")
    
    # Calculate multi-year statistics
    annual_mean_stack = np.stack(annual_mean_list, axis=0)
    annual_max_stack = np.stack(annual_max_list, axis=0)
    
    # Mean across years
    mean_precip = np.nanmean(annual_mean_stack, axis=0)
    max_precip = np.nanmax(annual_max_stack, axis=0)
    
    return mean_precip, max_precip, lat, lon, years_loaded

def cap_outliers(data, percentile=99.99):
    """Cap extreme values at specified percentile to handle outliers."""
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return data
    
    cap_value = np.percentile(valid_data, percentile)
    return np.where(data > cap_value, cap_value, data)

def calculate_latitudinal_stats(data, lat, mask=None):
    """Calculate latitudinal mean and standard deviation with optional masking."""
    if mask is not None:
        # Apply mask (set masked values to NaN)
        masked_data = np.where(mask, np.nan, data)
    else:
        masked_data = data
    
    # Calculate mean and std across longitude for each latitude
    lat_mean = np.nanmean(masked_data, axis=1)
    lat_std = np.nanstd(masked_data, axis=1)
    
    # Also calculate the number of valid points per latitude
    lat_count = np.sum(~np.isnan(masked_data), axis=1)
    
    return lat_mean, lat_std, lat_count

def create_academic_colormap(base_cmap='BrBG', n_colors=256):
    """Create academic-quality colormap based on BrBG."""
    if base_cmap == 'BrBG':
        # Brown-Blue-Green: excellent for precipitation data
        return plt.cm.BrBG_r  # Reverse so brown=low, blue/green=high
    elif base_cmap == 'precipitation':
        # Custom precipitation colormap
        colors = ['#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', 
                 '#C7EAE5', '#80CDC1', '#35978F', '#01665E']
        return LinearSegmentedColormap.from_list('precip_academic', colors, N=n_colors)
    else:
        return plt.cm.get_cmap(base_cmap)

def create_side_by_side_layout():
    """Create side-by-side layout with shared latitude axes."""
    # Set up the figure with academic paper quality
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5
    })
    
    fig = plt.figure(figsize=(12, 10))
    
    # Create subplot arrangement: 3 rows x 3 columns
    # Each row: [Map, Line plot, Colorbar]
    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 0.8, 0.05], height_ratios=[1, 1, 1], 
                         hspace=0.4, wspace=0.02)
    
    # Map subplots (left column)
    ax_maps = []
    ax_maps.append(fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()))
    ax_maps.append(fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()))
    ax_maps.append(fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree()))
    
    # Line plot subplots (middle column) - these will share y-axis with maps
    ax_lines = []
    ax_lines.append(fig.add_subplot(gs[0, 1]))
    ax_lines.append(fig.add_subplot(gs[1, 1]))
    ax_lines.append(fig.add_subplot(gs[2, 1]))
    
    return fig, ax_maps, ax_lines, gs

def plot_map_with_features(ax, data, lat, lon, title, cmap, vmin, vmax, mask=None):
    """Plot data on map with geographic features and advanced masking."""
    # Apply mask if provided
    if mask is not None:
        plot_data = np.where(mask, np.nan, data)
    else:
        plot_data = data
    
    # Create the map with higher resolution contours
    levels = np.linspace(vmin, vmax, 50)
    im = ax.contourf(lon, lat, plot_data, levels=levels, cmap=cmap, 
                    vmin=vmin, vmax=vmax, extend='both', transform=ccrs.PlateCarree())
    
    # Add geographic features with academic styling
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black', alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, color='gray', alpha=0.6)
    
    # Add subtle land/ocean background only where not masked
    if mask is not None:
        ax.add_feature(cfeature.LAND, alpha=0.2, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.1, facecolor='lightblue')
    
    # Set global extent with full world coverage
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Add refined gridlines with full latitude range (no automatic labels to avoid duplicates)
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.4, color='gray',
                     xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-90, 91, 30))
    
    return im

def plot_latitudinal_variation_with_uncertainty(ax, lat, lat_mean, lat_std, lat_count, title, color, xlabel):
    """Plot latitudinal variation with mean ± standard deviation."""
    # Remove NaN values
    valid_mask = ~np.isnan(lat_mean)
    valid_lat = lat[valid_mask]
    valid_mean = lat_mean[valid_mask]
    valid_std = lat_std[valid_mask]
    valid_count = lat_count[valid_mask]
    
    if len(valid_mean) == 0:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
               ha='center', va='center', fontsize=11, style='italic', color='red')
        return
    
    # Plot uncertainty band (mean ± std) first (background)
    lower_bound = valid_mean - valid_std
    upper_bound = valid_mean + valid_std
    
    ax.fill_betweenx(valid_lat, lower_bound, upper_bound, 
                     color=color, alpha=0.3, label='±1σ', zorder=1)
    
    # Plot mean line on top (more prominent)
    ax.plot(valid_mean, valid_lat, color=color, linewidth=3.0, 
           label='Mean', alpha=1.0, zorder=3)
    
    # Enhanced grid and styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Set latitude range to match global maps exactly (full world)
    ax.set_ylim(-90, 90)
    
    # Add key latitude lines at the same positions as map gridlines
    for lat_line in [-90, -60, -30, 0, 30, 60, 90]:
        ax.axhline(y=lat_line, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add zero line for precipitation if data crosses zero
    if np.any(valid_mean <= 0) and np.any(valid_mean > 0):
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Set reasonable x-axis limits based on data
    if len(valid_mean) > 0:
        data_min = np.min(lower_bound)
        data_max = np.max(upper_bound)
        data_range = data_max - data_min
        ax.set_xlim(data_min - 0.05*data_range, data_max + 0.05*data_range)
    
    # Add legend
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # Improve tick formatting
    ax.tick_params(axis='both', which='major', labelsize=9)

def main():
    """Main function to create the side-by-side precipitation analysis figure."""
    parser = argparse.ArgumentParser(description='Create side-by-side precipitation analysis figure')
    
    parser.add_argument('--percentile-file', 
                       default='data/processed/precipitation_percentiles.nc',
                       help='Precipitation percentile file')
    parser.add_argument('--etccdi-dir', 
                       default='data/processed/etccdi_indices',
                       help='Directory containing ETCCDI indices')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year for analysis (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--output-file', 
                       default='side_by_side_precipitation_analysis.png',
                       help='Output figure file')
    parser.add_argument('--mask', choices=['land', 'ocean', 'none'], default='land',
                       help='Mask option: land, ocean, or none (default: land)')
    parser.add_argument('--colormap', choices=['BrBG', 'precipitation'], default='BrBG',
                       help='Colormap choice (default: BrBG)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI for publication quality (default: 300)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SIDE-BY-SIDE PRECIPITATION ANALYSIS FIGURE GENERATOR")
    print("="*80)
    print(f"Percentile file: {args.percentile_file}")
    print(f"ETCCDI directory: {args.etccdi_dir}")
    print(f"Analysis period: {args.start_year}-{args.end_year}")
    print(f"Mask option: {args.mask}")
    print(f"Colormap: {args.colormap}")
    print(f"Output file: {args.output_file}")
    print("="*80)
    
    # Load data
    print("Loading precipitation percentiles...")
    try:
        p90_data, lat, lon = load_precipitation_percentiles(args.percentile_file)
        print(f"✅ Loaded P90 data: {p90_data.shape}")
        
        # Create dummy dataset for masking
        dummy_ds = xr.Dataset(coords={'latitude': lat, 'longitude': lon})
        
    except Exception as e:
        print(f"❌ Error loading percentiles: {e}")
        return 1
    
    print("Loading multi-year precipitation statistics...")
    try:
        mean_precip, max_precip, lat_etccdi, lon_etccdi, years_loaded = load_multi_year_precipitation_stats(
            args.etccdi_dir, args.start_year, args.end_year
        )
        print(f"✅ Loaded precipitation stats from {len(years_loaded)} years")
    except Exception as e:
        print(f"❌ Error loading ETCCDI data: {e}")
        return 1
    
    # Ensure coordinate consistency
    if not (np.allclose(lat, lat_etccdi, atol=1e-6) and np.allclose(lon, lon_etccdi, atol=1e-6)):
        print("⚠️  Warning: Coordinate mismatch between datasets")
    
    # Create advanced land/ocean mask
    print("Creating advanced land/ocean mask...")
    try:
        if args.mask == 'land':
            mask = create_land_ocean_mask(dummy_ds, 'ocean')  # Show ocean (mask land)
            mask_label = "Land Masked (Ocean Shown)"
        elif args.mask == 'ocean':
            mask = create_land_ocean_mask(dummy_ds, 'land')   # Show land (mask ocean)
            mask_label = "Ocean Masked (Land Shown)"
        else:
            mask = create_land_ocean_mask(dummy_ds, 'both')   # Show both
            mask_label = "No Masking"
            
        print(f"✅ Created advanced mask: {mask_label}")
    except Exception as e:
        print(f"⚠️  Warning: Could not create advanced mask, using simple fallback: {e}")
        mask = None
        mask_label = "No Masking (Fallback)"
    
    # Cap outliers
    print("Processing data and capping outliers...")
    p90_capped = cap_outliers(p90_data.values)
    mean_capped = cap_outliers(mean_precip)
    max_capped = cap_outliers(max_precip)
    
    print(f"P90 range: {np.nanmin(p90_capped):.2f} - {np.nanmax(p90_capped):.2f} mm/day")
    print(f"Mean range: {np.nanmin(mean_capped):.1f} - {np.nanmax(mean_capped):.1f} mm/year")
    print(f"Max range: {np.nanmin(max_capped):.1f} - {np.nanmax(max_capped):.1f} mm/day")
    
    # Calculate latitudinal statistics
    print("Calculating latitudinal statistics...")
    lat_p90_mean, lat_p90_std, lat_p90_count = calculate_latitudinal_stats(p90_capped, lat, mask)
    lat_mean_mean, lat_mean_std, lat_mean_count = calculate_latitudinal_stats(mean_capped, lat, mask)
    lat_max_mean, lat_max_std, lat_max_count = calculate_latitudinal_stats(max_capped, lat, mask)
    
    # Create figure
    print("Creating side-by-side figure...")
    fig, ax_maps, ax_lines, gs = create_side_by_side_layout()
    
    # Create academic colormap
    precip_cmap = create_academic_colormap(args.colormap)
    
    # Calculate masked data ranges for consistent colorbars and actual ranges
    if mask is not None:
        p90_masked, p90_vmin, p90_vmax, p90_count = apply_mask_and_adjust_colorbar(p90_capped, mask)
        mean_masked, mean_vmin, mean_vmax, mean_count = apply_mask_and_adjust_colorbar(mean_capped, mask)
        max_masked, max_vmin, max_vmax, max_count = apply_mask_and_adjust_colorbar(max_capped, mask)
        
        # Calculate actual data ranges
        p90_actual_min = np.nanmin(p90_capped[mask])
        p90_actual_max = np.nanmax(p90_capped[mask])
        mean_actual_min = np.nanmin(mean_capped[mask])
        mean_actual_max = np.nanmax(mean_capped[mask])
        max_actual_min = np.nanmin(max_capped[mask])
        max_actual_max = np.nanmax(max_capped[mask])
    else:
        p90_vmin, p90_vmax = np.nanmin(p90_capped), np.nanmax(p90_capped)
        mean_vmin, mean_vmax = np.nanmin(mean_capped), np.nanmax(mean_capped)
        max_vmin, max_vmax = np.nanmin(max_capped), np.nanmax(max_capped)
        p90_count = mean_count = max_count = "All"
        
        p90_actual_min, p90_actual_max = p90_vmin, p90_vmax
        mean_actual_min, mean_actual_max = mean_vmin, mean_vmax
        max_actual_min, max_actual_max = max_vmin, max_vmax
    
    # Plot Row 1: P90 percentile
    print("Plotting P90 percentile...")
    im1 = plot_map_with_features(ax_maps[0], p90_capped, lat, lon, 
                                f'(a) Average 90th Percentile Precipitation', 
                                precip_cmap, p90_vmin, p90_vmax, mask)
    
    plot_latitudinal_variation_with_uncertainty(ax_lines[0], lat, lat_p90_mean, lat_p90_std, lat_p90_count,
                                              '(b) Latitudinal Variation', 
                                              '#8C510A', 'P90 (mm/day)')
    
    # Plot Row 2: Mean precipitation
    print("Plotting mean precipitation...")
    im2 = plot_map_with_features(ax_maps[1], mean_capped, lat, lon, 
                                f'(c) Annual Mean Precipitation ({args.start_year}-{args.end_year})', 
                                precip_cmap, mean_vmin, mean_vmax, mask)
    
    plot_latitudinal_variation_with_uncertainty(ax_lines[1], lat, lat_mean_mean, lat_mean_std, lat_mean_count,
                                              '(d) Latitudinal Variation', 
                                              '#35978F', 'Mean (mm/year)')
    
    # Plot Row 3: Max precipitation
    print("Plotting max precipitation...")
    im3 = plot_map_with_features(ax_maps[2], max_capped, lat, lon, 
                                f'(e) Annual Max Precipitation ({args.start_year}-{args.end_year})', 
                                precip_cmap, max_vmin, max_vmax, mask)
    
    plot_latitudinal_variation_with_uncertainty(ax_lines[2], lat, lat_max_mean, lat_max_std, lat_max_count,
                                              '(f) Latitudinal Variation', 
                                              '#01665E', 'Max (mm/day)')
    
    # Add colorbars with custom tick formatting
    print("Adding colorbars...")
    
    # P90 colorbar - set specific ticks for better spacing
    cbar1_ax = fig.add_subplot(gs[0, 2])
    
    # Create normalized colorbar with extend='both'
    norm1 = mcolors.Normalize(vmin=p90_vmin, vmax=p90_vmax)
    dummy1 = plt.cm.ScalarMappable(norm=norm1, cmap=precip_cmap)
    cbar1 = fig.colorbar(dummy1, cax=cbar1_ax, extend='both')
    
    # Comprehensive colorbar label
    cbar1_label = f'mm/day\nColor: 0.5–99.5 percentile\nActual: {p90_actual_min:.1f}–{p90_actual_max:.1f}'
    cbar1.set_label(cbar1_label, fontsize=8, fontweight='bold', linespacing=1.2)
    
    # Set ticks strictly within [vmin, vmax]
    p90_ticks = np.linspace(p90_vmin, p90_vmax, 5)
    cbar1.set_ticks(p90_ticks)
    cbar1.set_ticklabels([f'{tick:.1f}' for tick in p90_ticks])
    cbar1.ax.tick_params(labelsize=8)
    
    # Mean precipitation colorbar - set specific ticks (0, 100, 200, 300, etc.)
    cbar2_ax = fig.add_subplot(gs[1, 2])
    
    # Create normalized colorbar with extend='both'
    norm2 = mcolors.Normalize(vmin=mean_vmin, vmax=mean_vmax)
    dummy2 = plt.cm.ScalarMappable(norm=norm2, cmap=precip_cmap)
    cbar2 = fig.colorbar(dummy2, cax=cbar2_ax, extend='both')
    
    # Comprehensive colorbar label
    cbar2_label = f'mm/year\nColor: 0.5–99.5 percentile\nActual: {mean_actual_min:.0f}–{mean_actual_max:.0f}'
    cbar2.set_label(cbar2_label, fontsize=8, fontweight='bold', linespacing=1.2)
    
    # Set ticks strictly within [vmin, vmax]
    mean_ticks = np.linspace(mean_vmin, mean_vmax, 5)
    cbar2.set_ticks(mean_ticks)
    cbar2.set_ticklabels([f'{tick:.0f}' for tick in mean_ticks])
    cbar2.ax.tick_params(labelsize=8)
    
    # Max precipitation colorbar - set specific ticks (0, 15, 30, etc.)
    cbar3_ax = fig.add_subplot(gs[2, 2])
    
    # Create normalized colorbar with extend='both'
    norm3 = mcolors.Normalize(vmin=max_vmin, vmax=max_vmax)
    dummy3 = plt.cm.ScalarMappable(norm=norm3, cmap=precip_cmap)
    cbar3 = fig.colorbar(dummy3, cax=cbar3_ax, extend='both')
    
    # Comprehensive colorbar label
    cbar3_label = f'mm/day\nColor: 0.5–99.5 percentile\nActual: {max_actual_min:.1f}–{max_actual_max:.1f}'
    cbar3.set_label(cbar3_label, fontsize=8, fontweight='bold', linespacing=1.2)
    
    # Set ticks strictly within [vmin, vmax]
    max_ticks = np.linspace(max_vmin, max_vmax, 5)
    cbar3.set_ticks(max_ticks)
    cbar3.set_ticklabels([f'{tick:.1f}' for tick in max_ticks])
    cbar3.ax.tick_params(labelsize=8)
    
    # Ensure perfect latitude alignment between maps and line plots
    print("Ensuring perfect latitude alignment...")
    
    # Define consistent latitude range and ticks for all plots (full world)
    lat_range = (-90, 90)
    lat_ticks = np.array([-90, -60, -30, 0, 30, 60, 90])
    
    for i in range(3):
        # Set exact same latitude limits for both map and line plot
        ax_maps[i].set_ylim(lat_range)
        ax_lines[i].set_ylim(lat_range)
        
        # Set exact same tick positions for both
        ax_maps[i].set_yticks(lat_ticks)
        ax_lines[i].set_yticks(lat_ticks)
        
        # Set consistent tick labels
        tick_labels = [f'{lat:.0f}°' for lat in lat_ticks]
        ax_maps[i].set_yticklabels(tick_labels)
        ax_lines[i].set_yticklabels(tick_labels)
        
        # Also set longitude ticks for maps to match gridlines
        lon_ticks = np.arange(-180, 181, 60)
        ax_maps[i].set_xticks(lon_ticks)
        ax_maps[i].set_xticklabels([f'{lon:.0f}°' for lon in lon_ticks])
        
        print(f"  Row {i+1}: Latitude range set to {lat_range}, ticks at {lat_ticks}")
    
    print("✅ Perfect latitude alignment achieved - 0° latitude line will appear at same y-position in both maps and line plots")
    
    # Adjust layout
    plt.tight_layout()
    
    print(f"Saving figure to {args.output_file}...")
    plt.savefig(args.output_file, dpi=args.dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✅ Figure saved successfully!")
    
    # Show summary statistics
    print("\n" + "="*80)
    print("LATITUDINAL STATISTICS SUMMARY")
    print("="*80)
    
    for i, (data_name, lat_mean, lat_std) in enumerate([
        ('P90 Precipitation', lat_p90_mean, lat_p90_std),
        ('Annual Mean Precipitation', lat_mean_mean, lat_mean_std),
        ('Annual Max Precipitation', lat_max_mean, lat_max_std)
    ]):
        valid_mask = ~np.isnan(lat_mean)
        if np.any(valid_mask):
            print(f"{data_name}:")
            print(f"  Latitudes with data: {np.sum(valid_mask)}/{len(lat_mean)}")
            print(f"  Mean across latitudes: {np.nanmean(lat_mean):.2f}")
            print(f"  Mean variability (std): {np.nanmean(lat_std):.2f}")
            
            # Find latitude of maximum and minimum
            max_idx = np.nanargmax(lat_mean)
            min_idx = np.nanargmin(lat_mean)
            print(f"  Maximum at: {lat[max_idx]:.1f}°N ({lat_mean[max_idx]:.2f})")
            print(f"  Minimum at: {lat[min_idx]:.1f}°N ({lat_mean[min_idx]:.2f})")
            print()
    
    print(f"🎯 Side-by-side analysis completed successfully!")
    print(f"📊 Publication-ready figure with uncertainty bands: {args.output_file}")
    
    plt.show()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())