#!/usr/bin/env python3
"""
Calculate precipitation percentiles for heatwave-precipitation analysis using ERA5 data.
Based on the successful approach from 01_calculate_percentiles.py and 03_calculate_RH_percentiles.py.

This script creates day-of-year climatology of precipitation percentiles (P10, P25, P50, P75, P90)
needed for drought and intense precipitation classification in heatwave analysis.

Precipitation data format:
- Variable: 'tp' (Total precipitation) 
- Units: meters (converted to mm by multiplying by 1000)
- Same spatial resolution as temperature data (0.25°)
"""

import sys
import argparse
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from functools import partial
import gc
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def load_precip_monthly_file(file_path):
    """Load a single precipitation monthly file."""
    try:
        ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
        return ds
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_daily_precipitation(precip_data):
    """Calculate daily precipitation from sub-daily data."""
    # Sum sub-daily precipitation to get daily totals
    daily_precip = precip_data.resample(valid_time='1D').sum()
    return daily_precip

def calculate_percentile_doy(data, times, percentile=10, window_days=15):
    """
    Calculate percentile for each day of year using a moving window.
    
    Parameters:
    - data: numpy array of shape (time, lat, lon) in mm
    - times: array of datetime objects
    - percentile: percentile to calculate (default 10)
    - window_days: window size around each day of year (default 15)
    """
    # Convert times to pandas datetime
    times_pd = pd.to_datetime(times)
    day_of_years = times_pd.dayofyear.values
    
    n_lat, n_lon = data.shape[1], data.shape[2]
    percentiles = np.full((366, n_lat, n_lon), np.nan)
    
    for doy in range(1, 367):
        # Create window around day of year (handles year boundaries)
        window_doys = []
        for offset in range(-window_days, window_days + 1):
            target_doy = doy + offset
            if target_doy <= 0:
                target_doy += 366
            elif target_doy > 366:
                target_doy -= 366
            window_doys.append(target_doy)
        
        # Find all days in the window across all years
        mask = np.isin(day_of_years, window_doys)
        
        if np.sum(mask) > 0:
            window_data = data[mask, :, :]
            # For precipitation, we want to handle zeros appropriately
            # Only calculate percentiles where there's some precipitation data
            percentiles[doy-1, :, :] = np.nanpercentile(window_data, percentile, axis=0)
    
    return percentiles

def process_spatial_chunk(chunk_info, years, precip_dir, percentiles=[10, 25, 50, 75, 90]):
    """Process a spatial chunk for precipitation percentile calculation."""
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    print(f"Processing precipitation chunk: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    # Collect all precipitation data for the chunk across all years
    all_precip_data = []
    all_times = []
    lat_coords = None
    lon_coords = None
    
    for year in years:
        for month in range(1, 13):
            file_path = Path(precip_dir) / f"era5_daily_{year}_{month:02d}.nc"
            
            if not file_path.exists():
                print(f"Warning: Precipitation file not found: {file_path}")
                continue
            
            try:
                ds = load_precip_monthly_file(file_path)
                if ds is None:
                    continue
                
                # Subset to chunk
                ds_chunk = ds.isel(
                    latitude=slice(lat_start, lat_end),
                    longitude=slice(lon_start, lon_end)
                )
                
                # Extract precipitation data
                if 'tp' in ds_chunk.variables:
                    precip_var = ds_chunk.tp
                elif 'total_precipitation' in ds_chunk.variables:
                    precip_var = ds_chunk.total_precipitation
                else:
                    print(f"Warning: No precipitation variable found in {file_path}")
                    ds.close()
                    continue
                
                # Handle time coordinate
                if 'valid_time' in precip_var.dims:
                    time_coord = 'valid_time'
                elif 'time' in precip_var.dims:
                    time_coord = 'time'
                else:
                    print(f"Warning: No time coordinate found in {file_path}")
                    ds.close()
                    continue
                
                # Calculate daily precipitation and convert to mm
                daily_precip = calculate_daily_precipitation(precip_var)
                daily_precip_mm = daily_precip * 1000.0  # Convert from m to mm
                
                # Store coordinates from first file
                if lat_coords is None:
                    lat_coords = daily_precip.latitude.values
                    lon_coords = daily_precip.longitude.values
                
                # Collect data
                all_precip_data.append(daily_precip_mm.values)
                all_times.extend(daily_precip.valid_time.values)
                
                # Clean up
                ds.close()
                del ds, ds_chunk, precip_var, daily_precip, daily_precip_mm
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    if not all_precip_data:
        print(f"No precipitation data found for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        return None
    
    # Combine all data
    combined_precip = np.concatenate(all_precip_data, axis=0)
    
    # Calculate percentiles for each requested percentile
    percentile_results = {}
    for p in percentiles:
        print(f"  Calculating P{p} for chunk lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
        percentile_results[f'precip_p{p}'] = calculate_percentile_doy(combined_precip, all_times, p)
    
    result = {
        'lat_bounds': (lat_start, lat_end),
        'lon_bounds': (lon_start, lon_end),
        'percentiles': percentile_results,
        'latitude': lat_coords,
        'longitude': lon_coords
    }
    
    # Clean up
    del combined_precip, all_precip_data
    gc.collect()
    
    print(f"Completed precipitation chunk: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    return result

def create_spatial_chunks(n_lat, n_lon, chunk_size_lat=50, chunk_size_lon=100):
    """Create spatial chunks for parallel processing."""
    chunks = []
    for lat_start in range(0, n_lat, chunk_size_lat):
        lat_end = min(lat_start + chunk_size_lat, n_lat)
        for lon_start in range(0, n_lon, chunk_size_lon):
            lon_end = min(lon_start + chunk_size_lon, n_lon)
            chunks.append((lat_start, lat_end, lon_start, lon_end))
    return chunks

def combine_chunks(chunk_results, full_lat, full_lon, output_file, percentiles, years):
    """Combine spatial chunks into full global arrays."""
    n_doy = 366
    
    # Initialize arrays for each percentile
    percentile_arrays = {}
    for p in percentiles:
        percentile_arrays[f'precip_p{p}'] = np.full((n_doy, len(full_lat), len(full_lon)), np.nan)
    
    # Fill arrays from chunk results
    for result in chunk_results:
        if result is None:
            continue
        
        lat_start, lat_end = result['lat_bounds']
        lon_start, lon_end = result['lon_bounds']
        
        for p in percentiles:
            var_name = f'precip_p{p}'
            percentile_arrays[var_name][:, lat_start:lat_end, lon_start:lon_end] = result['percentiles'][var_name]
    
    # Create dataset
    doy_coord = np.arange(1, 367)
    
    ds = xr.Dataset({
        var_name: (['dayofyear', 'latitude', 'longitude'], percentile_arrays[var_name])
        for var_name in percentile_arrays.keys()
    }, coords={
        'dayofyear': doy_coord,
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add attributes
    for p in percentiles:
        var_name = f'precip_p{p}'
        ds[var_name].attrs = {
            'long_name': f'{p}th percentile of daily precipitation',
            'units': 'mm',
            'description': f'Calculated from {years[0]}-{years[-1]} using 15-day window around each day of year',
            'baseline_period': f'{years[0]}-{years[-1]}',
            'calculation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_variable': 'ERA5 total precipitation (tp)',
            'original_units': 'meters (converted to mm)'
        }
    
    # Add global attributes
    ds.attrs = {
        'title': 'ERA5 Precipitation Percentiles by Day of Year',
        'description': 'Climatological percentiles of daily precipitation for heatwave-precipitation analysis',
        'source': 'Calculated from ERA5 total precipitation data',
        'climatology_period': f'{years[0]}-{years[-1]}',
        'percentiles': percentiles,
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'creator': 'ERA5 Precipitation Percentile Calculator',
        'methodology': 'Day-of-year climatology with 15-day moving window',
        'spatial_resolution': '0.25 degrees',
        'temporal_resolution': 'daily'
    }
    
    # Save file
    print("Saving precipitation percentile file...")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use compression for efficient storage
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} 
               for var in ds.data_vars}
    ds.to_netcdf(output_file, encoding=encoding)
    
    print(f"Precipitation percentile file saved: {output_file}")
    
    return output_file

def main():
    """Main function for precipitation percentile calculation."""
    parser = argparse.ArgumentParser(description='Calculate precipitation percentiles for heatwave-precipitation analysis')
    
    parser.add_argument('--precip-dir', default='/data/climate/disk1/datasets/era5',
                       help='Directory containing precipitation files')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year for climatology (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2000,
                       help='End year for climatology (default: 2000)')
    parser.add_argument('--output-file', default='data/processed/precipitation_percentiles.nc',
                       help='Output percentile file')
    parser.add_argument('--percentiles', nargs='+', type=int, default=[10, 25, 50, 75, 90, 95],
                       help='Percentiles to calculate (default: 10 25 50 75 90 95)')
    parser.add_argument('--n-processes', type=int, default=48,
                       help='Number of processes to use (default: 24)')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size (default: 50)')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size (default: 100)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ERA5 PRECIPITATION PERCENTILE CALCULATION")
    print("="*80)
    print(f"Precipitation directory: {args.precip_dir}")
    print(f"Climatology period: {args.start_year}-{args.end_year}")
    print(f"Percentiles: {args.percentiles}")
    print(f"Output file: {args.output_file}")
    print(f"Processes: {args.n_processes}")
    print(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    print("="*80)
    
    # Validate input directory
    precip_dir = Path(args.precip_dir)
    if not precip_dir.exists():
        raise ValueError(f"Precipitation directory does not exist: {precip_dir}")
    
    # Check if some precipitation files exist
    years = list(range(args.start_year, args.end_year + 1))
    sample_files = list(precip_dir.glob(f'era5_daily_{args.start_year}_*.nc'))
    if not sample_files:
        raise ValueError(f"No precipitation files found for year {args.start_year} in {precip_dir}")
    
    print(f"Found sample files: {len(sample_files)} for year {args.start_year}")
    
    # Get grid dimensions from first available file
    sample_file = sample_files[0]
    print(f"Reading grid dimensions from: {sample_file}")
    sample_ds = xr.open_dataset(sample_file)
    full_lat = sample_ds.latitude.values
    full_lon = sample_ds.longitude.values
    sample_ds.close()
    
    print(f"Grid dimensions: {len(full_lat)} x {len(full_lon)}")
    
    # Create spatial chunks
    spatial_chunks = create_spatial_chunks(
        len(full_lat), len(full_lon),
        args.chunk_size_lat, args.chunk_size_lon
    )
    
    print(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Estimate memory usage
    estimated_memory_per_process = 2.5  # GB per process (conservative estimate)
    total_estimated_memory = estimated_memory_per_process * args.n_processes
    
    print(f"Estimated memory usage: {total_estimated_memory:.1f} GB")
    if total_estimated_memory > 600:  # Conservative limit
        print("WARNING: High memory usage estimated. Consider reducing --n-processes if needed.")
    
    # Process chunks in parallel
    print(f"\nStarting parallel processing with {args.n_processes} processes...")
    
    process_func = partial(
        process_spatial_chunk,
        years=years,
        precip_dir=args.precip_dir,
        percentiles=args.percentiles
    )
    
    try:
        with mp.Pool(args.n_processes) as pool:
            chunk_results = pool.map(process_func, spatial_chunks)
        
        print("\nCombining chunks...")
        output_file = combine_chunks(chunk_results, full_lat, full_lon, 
                                   args.output_file, args.percentiles, years)
        
        print("\n" + "="*80)
        print("PRECIPITATION PERCENTILE CALCULATION COMPLETED!")
        print("="*80)
        print(f"Output file: {output_file}")
        
        # Verify output
        if output_file.exists():
            print("Verifying output file...")
            test_ds = xr.open_dataset(output_file)
            print(f"Output variables: {list(test_ds.data_vars)}")
            print(f"Day of year range: {test_ds.dayofyear.min().values} to {test_ds.dayofyear.max().values}")
            print(f"Spatial dimensions: {test_ds.dims}")
            
            # Show some statistics
            for var in test_ds.data_vars:
                data_min = test_ds[var].min().values
                data_max = test_ds[var].max().values
                data_mean = test_ds[var].mean().values
                print(f"{var}: min={data_min:.3f}, max={data_max:.3f}, mean={data_mean:.3f} mm")
            
            test_ds.close()
            print("✅ Output file verified successfully!")
        else:
            print("❌ Output file was not created!")
            return 1
            
    except Exception as e:
        print(f"Error calculating percentiles: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
