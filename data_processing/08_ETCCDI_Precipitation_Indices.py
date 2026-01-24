#!/usr/bin/env python3
"""
Calculate ETCCDI Precipitation Indices for Climate Change Analysis

Implements Expert Team on Climate Change Detection and Indices (ETCCDI) precipitation metrics:
- PRCPTOT: Annual precipitation from wet days (>1mm)
- R95p: Annual precipitation from very wet days (>P95)
- R95pTOT: Annual contribution from very wet days (R95p/PRCPTOT)
- WD50: Number of wettest days that constitute 50% of annual precipitation

References:
- Zhang et al. (2011): Indices for monitoring changes in extremes
- Peterson & Manton (2008): Climate change indices
- Alexander et al. (2006): Global observed changes in daily climate extremes
- Klein Tank & Können (2003): Trends in indices of daily temperature and precipitation extremes

Based on 1961-1990 climatological period for P95 calculation (following Klein Tank & Können, 2003).
Optimized for high-performance computing with parallel processing.
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
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_precipitation_data_year(year, precip_dir):
    """
    Load all precipitation data for a given year.
    
    Args:
        year: Target year
        precip_dir: Directory containing precipitation files
        
    Returns:
        precip_data: Daily precipitation array [time, lat, lon] in mm
        dates: Corresponding dates
    """
    precip_dir = Path(precip_dir)
    
    precip_data_list = []
    dates_list = []
    
    for month in range(1, 13):
        file_path = precip_dir / f"era5_daily_{year}_{month:02d}.nc"
        
        if not file_path.exists():
            logger.warning(f"Precipitation file not found: {file_path}")
            continue
        
        try:
            ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
            
            # Extract precipitation data and convert to mm/day
            if 'tp' in ds.variables:
                precip_var = ds.tp * 1000.0  # Convert from m to mm
            else:
                raise ValueError(f"No precipitation variable found in {file_path}")
            
            # Calculate daily precipitation (sum sub-daily if needed)
            daily_precip = precip_var.resample(valid_time='1D').sum()
            
            precip_data_list.append(daily_precip.values)
            dates_list.extend(pd.to_datetime(daily_precip.valid_time.values))
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if not precip_data_list:
        return None, None
    
    combined_precip = np.concatenate(precip_data_list, axis=0)
    combined_dates = pd.to_datetime(dates_list)
    
    return combined_precip, combined_dates

def load_precipitation_percentiles(percentile_file):
    """Load P95 precipitation percentiles for wet day threshold."""
    percentile_file = Path(percentile_file)
    
    if not percentile_file.exists():
        raise ValueError(f"Percentile file not found: {percentile_file}")
    
    ds = xr.open_dataset(percentile_file)
    
    if 'precip_p95' not in ds.data_vars:
        raise ValueError("P95 precipitation percentiles not found in file")
    
    # Calculate annual mean P95 (for wet days only)
    p95_annual = ds.precip_p95.mean(dim='dayofyear')
    
    ds.close()
    
    return p95_annual

def calculate_etccdi_indices(precip_data, dates, p95_threshold, wet_day_threshold=1.0):
    """
    Calculate ETCCDI precipitation indices for a single location and year.
    
    Args:
        precip_data: Daily precipitation array for one year [time]
        dates: Corresponding dates
        p95_threshold: P95 threshold for very wet days (mm/day)
        wet_day_threshold: Minimum precipitation for wet day (mm/day, default 1.0)
        
    Returns:
        Dictionary with ETCCDI indices
    """
    # Handle NaN values
    valid_mask = ~np.isnan(precip_data)
    if np.sum(valid_mask) == 0:
        return {
            'PRCPTOT': np.nan,
            'R95p': np.nan,
            'R95pTOT': np.nan,
            'WD50': np.nan,
            'wet_days': np.nan,
            'very_wet_days': np.nan,
            'max_daily_precip': np.nan
        }
    
    valid_precip = precip_data[valid_mask]
    
    # 1. PRCPTOT: Annual precipitation from wet days (>1mm)
    wet_days_mask = valid_precip >= wet_day_threshold
    wet_day_precip = valid_precip[wet_days_mask]
    PRCPTOT = np.sum(wet_day_precip)
    
    # 2. R95p: Annual precipitation from very wet days (>P95)
    if not np.isnan(p95_threshold) and p95_threshold > 0:
        very_wet_mask = valid_precip >= p95_threshold
        very_wet_precip = valid_precip[very_wet_mask]
        R95p = np.sum(very_wet_precip)
        very_wet_days = np.sum(very_wet_mask)
    else:
        R95p = np.nan
        very_wet_days = 0
    
    # 3. R95pTOT: Annual contribution from very wet days (R95p/PRCPTOT)
    if PRCPTOT > 0 and not np.isnan(R95p):
        R95pTOT = R95p / PRCPTOT
    else:
        R95pTOT = np.nan
    
    # 4. WD50: Number of wettest days that constitute 50% of annual precipitation
    if PRCPTOT > 0:
        # Sort precipitation in descending order
        sorted_precip = np.sort(valid_precip)[::-1]
        cumulative_precip = np.cumsum(sorted_precip)
        
        # Find number of days to reach 50% of total
        target_precip = PRCPTOT * 0.5
        wd50_idx = np.where(cumulative_precip >= target_precip)[0]
        WD50 = wd50_idx[0] + 1 if len(wd50_idx) > 0 else len(valid_precip)
    else:
        WD50 = np.nan
    
    # Additional metrics
    wet_days = np.sum(wet_days_mask)
    max_daily_precip = np.max(valid_precip) if len(valid_precip) > 0 else np.nan
    
    return {
        'PRCPTOT': PRCPTOT,
        'R95p': R95p,
        'R95pTOT': R95pTOT,
        'WD50': WD50,
        'wet_days': wet_days,
        'very_wet_days': very_wet_days,
        'max_daily_precip': max_daily_precip
    }

def process_spatial_chunk_etccdi(chunk_info, year, precip_dir, p95_thresholds, wet_day_threshold=1.0):
    """
    Process ETCCDI indices for a spatial chunk.
    
    Args:
        chunk_info: (lat_start, lat_end, lon_start, lon_end)
        year: Target year
        precip_dir: Directory containing precipitation files
        p95_thresholds: P95 threshold array [lat_chunk, lon_chunk]
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        Dictionary with ETCCDI index grids for the chunk
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing ETCCDI chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    # Load precipitation data for the year
    precip_data, dates = load_precipitation_data_year(year, precip_dir)
    
    if precip_data is None:
        logger.warning(f"No precipitation data found for year {year}")
        return None
    
    # Subset to spatial chunk
    precip_chunk = precip_data[:, lat_start:lat_end, lon_start:lon_end]
    n_time, n_lat, n_lon = precip_chunk.shape
    
    # Initialize output grids with explicit dtypes
    indices = {
        'PRCPTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95p': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95pTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'WD50': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'very_wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'max_daily_precip': np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    }
    
    # Process each pixel in the chunk
    for i in range(n_lat):
        for j in range(n_lon):
            pixel_precip = precip_chunk[:, i, j]
            pixel_p95 = p95_thresholds[i, j] if p95_thresholds is not None else np.nan
            
            # Calculate ETCCDI indices for this pixel
            pixel_indices = calculate_etccdi_indices(pixel_precip, dates, pixel_p95, wet_day_threshold)
            
            # Store results
            for index_name, value in pixel_indices.items():
                indices[index_name][i, j] = value
    
    logger.info(f"Completed ETCCDI chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    return {
        'chunk_bounds': chunk_info,
        'year': year,
        'indices': indices
    }

def combine_etccdi_results(chunk_results, full_lat, full_lon, year, output_dir):
    """
    Combine spatial chunks into full global arrays and save results.
    
    Args:
        chunk_results: List of chunk processing results
        full_lat: Full latitude array
        full_lon: Full longitude array
        year: Target year
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize full grids with explicit dtypes
    n_lat, n_lon = len(full_lat), len(full_lon)
    
    full_indices = {
        'PRCPTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95p': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95pTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'WD50': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'very_wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'max_daily_precip': np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    }
    
    # Fill grids from chunk results
    for result in chunk_results:
        if result is None:
            continue
        
        lat_start, lat_end, lon_start, lon_end = result['chunk_bounds']
        
        for index_name in full_indices.keys():
            full_indices[index_name][lat_start:lat_end, lon_start:lon_end] = result['indices'][index_name]
    
    # Create dataset
    ds = xr.Dataset({
        index_name: (['latitude', 'longitude'], full_indices[index_name])
        for index_name in full_indices.keys()
    }, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add variable attributes
    ds['PRCPTOT'].attrs = {
        'long_name': 'Annual precipitation from wet days',
        'standard_name': 'PRCPTOT',
        'units': 'mm',
        'description': 'Total annual precipitation from days with precipitation >= 1 mm',
        'wet_day_threshold': '1.0 mm/day',
        'reference': 'ETCCDI (Zhang et al., 2011)'
    }
    
    ds['R95p'].attrs = {
        'long_name': 'Annual precipitation from very wet days',
        'standard_name': 'R95p',
        'units': 'mm',
        'description': 'Total annual precipitation from days exceeding 95th percentile',
        'percentile_period': '1980-2000',
        'reference': 'ETCCDI (Zhang et al., 2011)'
    }
    
    ds['R95pTOT'].attrs = {
        'long_name': 'Annual contribution from very wet days',
        'standard_name': 'R95pTOT',
        'units': 'fraction',
        'description': 'Fraction of annual precipitation from very wet days (R95p/PRCPTOT)',
        'valid_range': [0.0, 1.0],
        'reference': 'ETCCDI (Zhang et al., 2011)'
    }
    
    ds['WD50'].attrs = {
        'long_name': 'Number of wettest days contributing 50% of annual precipitation',
        'standard_name': 'WD50',
        'units': 'days',
        'description': 'Number of wettest days needed to reach 50% of annual precipitation total',
        'methodology': 'Descending sort of daily precipitation with cumulative sum',
        'reference': 'Custom index for precipitation concentration analysis'
    }
    
    ds['wet_days'].attrs = {
        'long_name': 'Number of wet days',
        'units': 'days',
        'description': 'Annual count of days with precipitation >= 1 mm',
        'wet_day_threshold': '1.0 mm/day'
    }
    
    ds['very_wet_days'].attrs = {
        'long_name': 'Number of very wet days',
        'units': 'days',
        'description': 'Annual count of days exceeding 95th percentile',
        'percentile_period': '1980-2000'
    }
    
    ds['max_daily_precip'].attrs = {
        'long_name': 'Maximum daily precipitation',
        'units': 'mm',
        'description': 'Maximum daily precipitation in the year'
    }
    
    # Add global attributes
    ds.attrs = {
        'title': 'ETCCDI Precipitation Indices',
        'description': 'Annual precipitation indices following ETCCDI recommendations',
        'source': 'Calculated from ERA5 total precipitation data',
        'year': year,
        'wet_day_threshold': '1.0 mm/day',
        'percentile_period': '1980-2000 (for P95 calculation)',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'creator': 'ETCCDI Precipitation Index Calculator',
        'references': 'Zhang et al. (2011), Peterson & Manton (2008), Alexander et al. (2006)',
        'institution': 'Climate Analysis',
        'spatial_resolution': '0.25 degrees'
    }
    
    # Save file
    output_file = output_dir / f'etccdi_precipitation_indices_{year}.nc'
    
    # Use compression for efficient storage
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} 
               for var in ds.data_vars}
    
    ds.to_netcdf(output_file, encoding=encoding)
    ds.close()
    
    logger.info(f"Saved ETCCDI indices for {year}: {output_file}")
    
    return output_file

def create_spatial_chunks(n_lat, n_lon, chunk_size_lat=50, chunk_size_lon=100):
    """Create spatial chunks for parallel processing."""
    chunks = []
    for lat_start in range(0, n_lat, chunk_size_lat):
        lat_end = min(lat_start + chunk_size_lat, n_lat)
        for lon_start in range(0, n_lon, chunk_size_lon):
            lon_end = min(lon_start + chunk_size_lon, n_lon)
            chunks.append((lat_start, lat_end, lon_start, lon_end))
    return chunks

def process_year_etccdi(year, precip_dir, percentile_file, output_dir, n_processes, 
                       chunk_size_lat, chunk_size_lon, wet_day_threshold):
    """
    Process ETCCDI indices for a complete year using parallel processing.
    
    Args:
        year: Target year
        precip_dir: Directory containing precipitation files
        percentile_file: Path to precipitation percentile file
        output_dir: Output directory
        n_processes: Number of processes for parallel computation
        chunk_size_lat: Latitude chunk size
        chunk_size_lon: Longitude chunk size
        wet_day_threshold: Minimum precipitation for wet day
        
    Returns:
        Path to output file
    """
    logger.info(f"Processing ETCCDI indices for year {year}")
    
    # Load P95 thresholds
    try:
        p95_global = load_precipitation_percentiles(percentile_file)
        logger.info(f"Loaded P95 thresholds from {percentile_file}")
    except Exception as e:
        logger.error(f"Could not load P95 thresholds: {e}")
        logger.info("Proceeding without P95 thresholds (R95p will be NaN)")
        p95_global = None
    
    # Get grid dimensions from sample file
    sample_file = None
    for month in range(1, 13):
        test_file = Path(precip_dir) / f"era5_daily_{year}_{month:02d}.nc"
        if test_file.exists():
            sample_file = test_file
            break
    
    if sample_file is None:
        raise ValueError(f"No precipitation files found for year {year}")
    
    sample_ds = xr.open_dataset(sample_file)
    full_lat = sample_ds.latitude.values
    full_lon = sample_ds.longitude.values
    sample_ds.close()
    
    logger.info(f"Grid dimensions: {len(full_lat)} x {len(full_lon)}")
    
    # Create spatial chunks
    spatial_chunks = create_spatial_chunks(
        len(full_lat), len(full_lon),
        chunk_size_lat, chunk_size_lon
    )
    
    logger.info(f"Created {len(spatial_chunks)} spatial chunks")
    
    # Process chunks in parallel
    logger.info(f"Processing with {n_processes} processes...")
    
    try:
        with mp.Pool(n_processes) as pool:
            if p95_global is not None:
                # Prepare chunk-specific P95 data
                chunk_args = []
                for lat_start, lat_end, lon_start, lon_end in spatial_chunks:
                    chunk_p95 = p95_global.values[lat_start:lat_end, lon_start:lon_end]
                    chunk_args.append(((lat_start, lat_end, lon_start, lon_end), year, precip_dir, chunk_p95, wet_day_threshold))
                
                logger.info(f"Processing {len(chunk_args)} chunks with P95 thresholds...")
                chunk_results = pool.starmap(process_spatial_chunk_etccdi_with_p95, chunk_args)
            else:
                # Process without P95 thresholds - create args for the basic function
                logger.info(f"Processing {len(spatial_chunks)} chunks without P95 thresholds...")
                chunk_args = []
                for chunk_info in spatial_chunks:
                    chunk_args.append((chunk_info, year, precip_dir, None, wet_day_threshold))
                
                chunk_results = pool.starmap(process_spatial_chunk_etccdi_with_p95, chunk_args)
                
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise
    
    # Combine results
    output_file = combine_etccdi_results(chunk_results, full_lat, full_lon, year, output_dir)
    
    return output_file

def process_spatial_chunk_etccdi_with_p95(chunk_info, year, precip_dir, p95_chunk, wet_day_threshold):
    """Process ETCCDI indices with chunk-specific P95 thresholds."""
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing ETCCDI chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    # Load precipitation data for the year and subset to chunk
    all_precip_data = []
    all_dates = []
    
    for month in range(1, 13):
        file_path = Path(precip_dir) / f"era5_daily_{year}_{month:02d}.nc"
        
        if not file_path.exists():
            continue
        
        try:
            ds = xr.open_dataset(file_path, chunks={'valid_time': 50})
            
            # Subset to chunk
            ds_chunk = ds.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end)
            )
            
            # Extract precipitation and convert to mm
            if 'tp' in ds_chunk.variables:
                precip_var = ds_chunk.tp * 1000.0
            else:
                ds.close()
                continue
            
            # Calculate daily precipitation
            daily_precip = precip_var.resample(valid_time='1D').sum()
            
            all_precip_data.append(daily_precip.values)
            all_dates.extend(pd.to_datetime(daily_precip.valid_time.values))
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if not all_precip_data:
        return None
    
    # Combine data
    combined_precip = np.concatenate(all_precip_data, axis=0)
    combined_dates = pd.to_datetime(all_dates)
    
    n_time, n_lat, n_lon = combined_precip.shape
    
    # Initialize output grids with explicit dtypes
    indices = {
        'PRCPTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95p': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'R95pTOT': np.full((n_lat, n_lon), np.nan, dtype=np.float64),
        'WD50': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'very_wet_days': np.full((n_lat, n_lon), np.nan, dtype=np.float64),  # Ensure float, not timedelta
        'max_daily_precip': np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    }
    
    # Process each pixel
    for i in range(n_lat):
        for j in range(n_lon):
            pixel_precip = combined_precip[:, i, j]
            pixel_p95 = p95_chunk[i, j] if p95_chunk is not None else np.nan
            
            # Calculate ETCCDI indices
            pixel_indices = calculate_etccdi_indices(pixel_precip, combined_dates, pixel_p95, wet_day_threshold)
            
            # Store results
            for index_name, value in pixel_indices.items():
                indices[index_name][i, j] = value
    
    logger.info(f"Completed ETCCDI chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    return {
        'chunk_bounds': chunk_info,
        'year': year,
        'indices': indices
    }

def main():
    """Main function for ETCCDI precipitation indices calculation."""
    parser = argparse.ArgumentParser(description='Calculate ETCCDI precipitation indices')
    
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year for analysis (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--precip-dir', default='/data/climate/disk1/datasets/era5',
                       help='Directory containing precipitation files')
    parser.add_argument('--percentile-file', default='data/processed/precipitation_percentiles.nc',
                       help='Precipitation percentile file (must contain P95)')
    parser.add_argument('--output-dir', default='data/processed/etccdi_indices',
                       help='Output directory for ETCCDI indices')
    parser.add_argument('--wet-day-threshold', type=float, default=1.0,
                       help='Minimum precipitation for wet day (mm/day, default: 1.0)')
    parser.add_argument('--n-processes', type=int, default=48,
                       help='Number of processes to use (default: 24)')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size (default: 50)')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size (default: 100)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("ETCCDI PRECIPITATION INDICES CALCULATION")
    logger.info("="*80)
    logger.info(f"Analysis period: {args.start_year}-{args.end_year}")
    logger.info(f"Precipitation directory: {args.precip_dir}")
    logger.info(f"Percentile file: {args.percentile_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wet day threshold: {args.wet_day_threshold} mm/day")
    logger.info(f"Processes: {args.n_processes}")
    logger.info(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    logger.info("="*80)
    
    # Validate directories
    precip_dir = Path(args.precip_dir)
    if not precip_dir.exists():
        raise ValueError(f"Precipitation directory does not exist: {precip_dir}")
    
    percentile_file = Path(args.percentile_file)
    if not percentile_file.exists():
        logger.warning(f"Percentile file not found: {percentile_file}")
        logger.warning("R95p and R95pTOT indices will be NaN")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each year
    years = list(range(args.start_year, args.end_year + 1))
    successful_years = []
    
    for year in years:
        logger.info(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        try:
            output_file = process_year_etccdi(
                year, args.precip_dir, args.percentile_file, args.output_dir,
                args.n_processes, args.chunk_size_lat, args.chunk_size_lon,
                args.wet_day_threshold
            )
            
            if output_file:
                successful_years.append(year)
                logger.info(f"✅ Completed year {year}")
            else:
                logger.error(f"❌ Failed to process year {year}")
                
        except Exception as e:
            logger.error(f"❌ Error processing year {year}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("ETCCDI CALCULATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully processed: {len(successful_years)}/{len(years)} years")
    logger.info(f"Years completed: {successful_years}")
    logger.info(f"Output directory: {args.output_dir}")
    
    if successful_years:
        logger.info("\nGenerated files:")
        for year in successful_years[:5]:  # Show first 5
            output_file = output_dir / f'etccdi_precipitation_indices_{year}.nc'
            if output_file.exists():
                logger.info(f"  - {output_file.name}")
        if len(successful_years) > 5:
            logger.info(f"  ... and {len(successful_years) - 5} more files")
        
        logger.info("\n🎯 ETCCDI indices calculated successfully!")
        logger.info("📊 Ready for visualization and climate analysis!")
        
        # Suggest next steps
        logger.info("\n📋 Next steps:")
        logger.info("1. Create multi-year climatology:")
        logger.info(f"   python create_etccdi_climatology.py --input-dir {args.output_dir}")
        logger.info("2. Run visualizations:")
        logger.info(f"   python visualizations/viz_08_etccdi_indices.py --etccdi-dir {args.output_dir}")
        logger.info("3. Analyze trends:")
        logger.info(f"   python analyze_etccdi_trends.py --etccdi-dir {args.output_dir}")
        
    else:
        logger.error("❌ No years processed successfully!")
        logger.error("Check precipitation data availability and file formats")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
