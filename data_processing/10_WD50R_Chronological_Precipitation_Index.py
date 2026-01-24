#!/usr/bin/env python3
"""
WD50R Chronological Precipitation Index Calculator

Calculates minimum consecutive days required to reach specified percentiles of annual precipitation.
This provides insights into precipitation event clustering and extreme event timing patterns.

Key Indices:
- WD50R: Minimum consecutive days to reach 50% of annual precipitation
- WD25R, WD75R, WD90R: Similar for other percentiles
- Start_day: Day of year when optimal window begins
- End_day: Day of year when optimal window ends
- Window_efficiency: Ratio of window sum to theoretical minimum

Algorithm:
For each possible starting day, find the shortest consecutive period that contains
the target percentage of annual precipitation. The global minimum becomes WDxxR.

Optimizations:
- Early termination when remaining days cannot reach target
- Vectorized operations where possible
- Regime-aware processing (skip very dry regions)
- Memory-efficient chunked processing

References:
- Novel methodology for precipitation concentration timing analysis
- Extends ETCCDI framework with temporal clustering metrics
- Applications in extreme event analysis and seasonal pattern detection

Optimized for high-performance computing with configurable analysis parameters.
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
from scipy import stats

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

def find_minimum_consecutive_window(daily_precip, target_fraction, wet_day_threshold=0.1):
    """
    Find minimum consecutive days to reach target percentage of annual precipitation.
    
    Args:
        daily_precip: Daily precipitation array for one year [time]
        target_fraction: Target fraction (e.g., 0.5 for 50%)
        wet_day_threshold: Minimum daily precipitation to consider (mm)
        
    Returns:
        min_window_length: Minimum consecutive days
        best_start_day: Starting day of year (1-based)
        best_end_day: Ending day of year (1-based)
        window_sum: Total precipitation in optimal window
    """
    # Calculate annual total and target
    annual_total = np.sum(daily_precip)
    target = annual_total * target_fraction
    
    if annual_total <= 0 or target <= 0:
        return np.nan, np.nan, np.nan, np.nan
    
    n_days = len(daily_precip)
    min_window_length = np.inf
    best_start = np.nan
    best_end = np.nan
    best_sum = np.nan
    
    # Try each possible starting day
    for start_day in range(n_days):
        cumulative_sum = 0
        
        # Optimization: Check if remaining days can possibly reach target
        remaining_total = np.sum(daily_precip[start_day:])
        if remaining_total < target:
            break  # No point trying later start dates
        
        # Accumulate from start_day until target is reached
        for end_day in range(start_day, n_days):
            cumulative_sum += daily_precip[end_day]
            
            if cumulative_sum >= target:
                window_length = end_day - start_day + 1
                
                if window_length < min_window_length:
                    min_window_length = window_length
                    best_start = start_day + 1  # Convert to 1-based day of year
                    best_end = end_day + 1      # Convert to 1-based day of year
                    best_sum = cumulative_sum
                
                break  # Found window for this start_day, move to next
    
    if min_window_length == np.inf:
        return np.nan, np.nan, np.nan, np.nan
    
    return min_window_length, best_start, best_end, best_sum

def classify_precipitation_regime_wd50r(annual_total, wd50r):
    """
    Classify precipitation regime based on annual total and WD50R characteristics.
    
    Args:
        annual_total: Annual precipitation total (mm)
        wd50r: WD50R value (days)
        
    Returns:
        regime: String classification
    """
    if np.isnan(annual_total) or annual_total < 50:
        return "HYPER_ARID"
    elif annual_total < 200:
        return "ARID"
    elif annual_total < 500:
        return "SEMI_ARID"
    elif not np.isnan(wd50r):
        if wd50r <= 5:
            return "EXTREME_CLUSTER"    # Very concentrated events
        elif wd50r <= 15:
            return "HIGH_CLUSTER"       # Moderately concentrated
        elif wd50r <= 30:
            return "MODERATE_CLUSTER"   # Some clustering
        elif wd50r <= 60:
            return "LOW_CLUSTER"        # Distributed patterns
        else:
            return "VERY_DISTRIBUTED"   # Highly distributed
    else:
        return "UNDEFINED"

def calculate_window_efficiency_metrics(daily_precip, percentiles, annual_total):
    """
    Calculate efficiency and clustering metrics for multiple percentiles.
    
    Args:
        daily_precip: Daily precipitation array
        percentiles: List of percentiles to analyze
        annual_total: Annual precipitation total
        
    Returns:
        Dictionary with efficiency metrics
    """
    if annual_total <= 0:
        return {f'efficiency_{p}': np.nan for p in percentiles}
    
    efficiency_metrics = {}
    
    for pct in percentiles:
        target = annual_total * (pct / 100.0)
        
        # Theoretical minimum: if all target precipitation fell on consecutive days
        sorted_precip = np.sort(daily_precip)[::-1]  # Descending order
        cumsum = np.cumsum(sorted_precip)
        theoretical_days = np.where(cumsum >= target)[0]
        
        if len(theoretical_days) > 0:
            theoretical_min = theoretical_days[0] + 1
        else:
            theoretical_min = len(daily_precip)
        
        # Actual WDxxR
        actual_wd, _, _, _ = find_minimum_consecutive_window(daily_precip, pct / 100.0)
        
        if not np.isnan(actual_wd) and theoretical_min > 0:
            efficiency = theoretical_min / actual_wd  # How close to optimal
        else:
            efficiency = np.nan
        
        efficiency_metrics[f'efficiency_{pct}'] = efficiency
    
    return efficiency_metrics

def calculate_wd50r_indices(precip_data, dates, percentiles=[25, 50, 75, 90], 
                           wet_day_threshold=0.1, min_annual_precip=50):
    """
    Calculate WDxxR indices for multiple percentiles for a single location and year.
    
    Args:
        precip_data: Daily precipitation array for one year [time]
        dates: Corresponding dates
        percentiles: List of percentiles to calculate (default: [25, 50, 75, 90])
        wet_day_threshold: Minimum daily precipitation (mm)
        min_annual_precip: Skip pixels with less than this annual total (mm)
        
    Returns:
        Dictionary with WDxxR indices
    """
    # Handle NaN values
    valid_mask = ~np.isnan(precip_data)
    if np.sum(valid_mask) == 0:
        return create_empty_wd50r_result(percentiles)
    
    valid_precip = precip_data[valid_mask]
    annual_total = np.sum(valid_precip)
    
    # Skip very dry pixels
    if annual_total < min_annual_precip:
        result = create_empty_wd50r_result(percentiles)
        result['annual_total'] = annual_total
        result['regime'] = "HYPER_ARID"
        return result
    
    # Initialize results
    results = {
        'annual_total': annual_total,
        'total_days': len(valid_precip),
        'wet_days': np.sum(valid_precip >= wet_day_threshold)
    }
    
    # Calculate WDxxR for each percentile
    for pct in percentiles:
        wd_length, start_day, end_day, window_sum = find_minimum_consecutive_window(
            valid_precip, pct / 100.0, wet_day_threshold
        )
        
        results[f'WD{pct}R'] = wd_length
        results[f'WD{pct}R_start'] = start_day
        results[f'WD{pct}R_end'] = end_day
        results[f'WD{pct}R_sum'] = window_sum
        
        # Calculate intensity (precipitation per day in optimal window)
        if not np.isnan(wd_length) and wd_length > 0:
            results[f'WD{pct}R_intensity'] = window_sum / wd_length
        else:
            results[f'WD{pct}R_intensity'] = np.nan
    
    # Classify regime based on WD50R
    wd50r = results.get('WD50R', np.nan)
    results['regime'] = classify_precipitation_regime_wd50r(annual_total, wd50r)
    
    # Calculate efficiency metrics
    efficiency_metrics = calculate_window_efficiency_metrics(
        valid_precip, percentiles, annual_total
    )
    results.update(efficiency_metrics)
    
    # Calculate clustering ratio (WD75R / WD25R)
    if 'WD75R' in results and 'WD25R' in results:
        wd75r = results['WD75R']
        wd25r = results['WD25R']
        if not np.isnan(wd75r) and not np.isnan(wd25r) and wd25r > 0:
            results['clustering_ratio'] = wd75r / wd25r
        else:
            results['clustering_ratio'] = np.nan
    
    # Calculate temporal spread
    if 'WD75R_start' in results and 'WD25R_start' in results:
        start_75 = results['WD75R_start']
        start_25 = results['WD25R_start']
        if not np.isnan(start_75) and not np.isnan(start_25):
            results['temporal_spread'] = abs(start_75 - start_25)
        else:
            results['temporal_spread'] = np.nan
    
    return results

def create_empty_wd50r_result(percentiles):
    """Create empty result dictionary for invalid pixels."""
    result = {
        'annual_total': np.nan,
        'total_days': 0,
        'wet_days': 0,
        'regime': "NO_DATA",
        'clustering_ratio': np.nan,
        'temporal_spread': np.nan
    }
    
    for pct in percentiles:
        result[f'WD{pct}R'] = np.nan
        result[f'WD{pct}R_start'] = np.nan
        result[f'WD{pct}R_end'] = np.nan
        result[f'WD{pct}R_sum'] = np.nan
        result[f'WD{pct}R_intensity'] = np.nan
        result[f'efficiency_{pct}'] = np.nan
    
    return result

def process_spatial_chunk_wd50r(chunk_info, year, precip_dir, percentiles, 
                               wet_day_threshold, min_annual_precip):
    """
    Process WD50R indices for a spatial chunk.
    
    Args:
        chunk_info: (lat_start, lat_end, lon_start, lon_end)
        year: Target year
        precip_dir: Directory containing precipitation files
        percentiles: List of percentiles to calculate
        wet_day_threshold: Minimum daily precipitation (mm)
        min_annual_precip: Skip pixels below this threshold (mm)
        
    Returns:
        Dictionary with WD50R index grids for the chunk
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing WD50R chunk: year={year}, percentiles={percentiles}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
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
    
    # Define output variables
    var_names = ['annual_total', 'total_days', 'wet_days', 'clustering_ratio', 'temporal_spread']
    var_names.extend([f'WD{pct}R' for pct in percentiles])
    var_names.extend([f'WD{pct}R_start' for pct in percentiles])
    var_names.extend([f'WD{pct}R_end' for pct in percentiles])
    var_names.extend([f'WD{pct}R_sum' for pct in percentiles])
    var_names.extend([f'WD{pct}R_intensity' for pct in percentiles])
    var_names.extend([f'efficiency_{pct}' for pct in percentiles])
    
    # Initialize output grids
    indices = {}
    for var_name in var_names:
        if var_name == 'regime':
            indices[var_name] = np.full((n_lat, n_lon), "", dtype='U20')
        else:
            indices[var_name] = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    
    # Add regime separately
    indices['regime'] = np.full((n_lat, n_lon), "", dtype='U20')
    
    # Process each pixel
    for i in range(n_lat):
        for j in range(n_lon):
            pixel_precip = combined_precip[:, i, j]
            
            # Calculate WD50R indices
            pixel_indices = calculate_wd50r_indices(
                pixel_precip, combined_dates, percentiles, 
                wet_day_threshold, min_annual_precip
            )
            
            # Store results
            for var_name in var_names:
                if var_name in pixel_indices:
                    indices[var_name][i, j] = pixel_indices[var_name]
            
            # Store regime separately
            indices['regime'][i, j] = pixel_indices.get('regime', 'NO_DATA')
    
    logger.info(f"Completed WD50R chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    return {
        'chunk_bounds': chunk_info,
        'year': year,
        'percentiles': percentiles,
        'indices': indices
    }

def combine_wd50r_results(chunk_results, full_lat, full_lon, year, percentiles, output_dir):
    """
    Combine spatial chunks into full global arrays and save results.
    
    Args:
        chunk_results: List of chunk processing results
        full_lat: Full latitude array
        full_lon: Full longitude array
        year: Target year
        percentiles: List of percentiles calculated
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get variable names from first successful result
    var_names = None
    for result in chunk_results:
        if result is not None:
            var_names = [k for k in result['indices'].keys() if k != 'regime']
            break
    
    if var_names is None:
        logger.error("No successful chunk results to combine")
        return None
    
    # Initialize full grids
    n_lat, n_lon = len(full_lat), len(full_lon)
    
    full_indices = {}
    for var_name in var_names:
        full_indices[var_name] = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    
    # Initialize regime grid separately
    full_regime = np.full((n_lat, n_lon), "", dtype='U20')
    
    # Fill grids from chunk results
    for result in chunk_results:
        if result is None:
            continue
        
        lat_start, lat_end, lon_start, lon_end = result['chunk_bounds']
        
        for var_name in var_names:
            if var_name in result['indices']:
                full_indices[var_name][lat_start:lat_end, lon_start:lon_end] = result['indices'][var_name]
        
        # Handle regime separately
        if 'regime' in result['indices']:
            full_regime[lat_start:lat_end, lon_start:lon_end] = result['indices']['regime']
    
    # Create dataset (exclude string variables from xarray)
    data_vars = {}
    for var_name in var_names:
        data_vars[var_name] = (['latitude', 'longitude'], full_indices[var_name])
    
    ds = xr.Dataset(data_vars, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add variable attributes
    for pct in percentiles:
        # Main WDxxR variables
        if f'WD{pct}R' in ds.data_vars:
            ds[f'WD{pct}R'].attrs = {
                'long_name': f'Minimum consecutive days for {pct}% of annual precipitation',
                'units': 'days',
                'description': f'Shortest consecutive period containing {pct}% of annual precipitation',
                'methodology': 'Sliding window optimization with early termination',
                'valid_range': [1, 365]
            }
        
        # Start day variables
        if f'WD{pct}R_start' in ds.data_vars:
            ds[f'WD{pct}R_start'].attrs = {
                'long_name': f'Start day of WD{pct}R window',
                'units': 'day of year',
                'description': f'Day of year when optimal {pct}% window begins',
                'valid_range': [1, 365]
            }
        
        # End day variables
        if f'WD{pct}R_end' in ds.data_vars:
            ds[f'WD{pct}R_end'].attrs = {
                'long_name': f'End day of WD{pct}R window',
                'units': 'day of year', 
                'description': f'Day of year when optimal {pct}% window ends',
                'valid_range': [1, 365]
            }
        
        # Window sum variables
        if f'WD{pct}R_sum' in ds.data_vars:
            ds[f'WD{pct}R_sum'].attrs = {
                'long_name': f'Precipitation sum in WD{pct}R window',
                'units': 'mm',
                'description': f'Total precipitation in optimal {pct}% window'
            }
        
        # Intensity variables  
        if f'WD{pct}R_intensity' in ds.data_vars:
            ds[f'WD{pct}R_intensity'].attrs = {
                'long_name': f'Average daily intensity in WD{pct}R window',
                'units': 'mm/day',
                'description': f'Mean daily precipitation in optimal {pct}% window'
            }
        
        # Efficiency variables
        if f'efficiency_{pct}' in ds.data_vars:
            ds[f'efficiency_{pct}'].attrs = {
                'long_name': f'Window efficiency for {pct}% threshold',
                'units': 'dimensionless',
                'description': f'Ratio of theoretical minimum to actual WD{pct}R',
                'valid_range': [0, 1],
                'interpretation': '1=perfect efficiency, 0=poor efficiency'
            }
    
    # Additional metrics
    if 'annual_total' in ds.data_vars:
        ds['annual_total'].attrs = {
            'long_name': 'Annual precipitation total',
            'units': 'mm',
            'description': 'Total annual precipitation'
        }
    
    if 'clustering_ratio' in ds.data_vars:
        ds['clustering_ratio'].attrs = {
            'long_name': 'Precipitation clustering ratio',
            'units': 'dimensionless',
            'description': 'Ratio of WD75R to WD25R',
            'interpretation': 'Higher values indicate more distributed patterns'
        }
    
    if 'temporal_spread' in ds.data_vars:
        ds['temporal_spread'].attrs = {
            'long_name': 'Temporal spread of precipitation windows',
            'units': 'days',
            'description': 'Difference between WD75R and WD25R start days',
            'interpretation': 'Seasonal separation of precipitation concentration'
        }
    
    # Add global attributes
    ds.attrs = {
        'title': 'WD50R Chronological Precipitation Indices',
        'description': 'Minimum consecutive days for precipitation percentage thresholds',
        'source': 'Calculated from ERA5 total precipitation data',
        'year': year,
        'percentiles': percentiles,
        'methodology': 'Sliding window optimization with early termination',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'creator': 'WD50R Chronological Index Calculator',
        'institution': 'Climate Analysis',
        'spatial_resolution': '0.25 degrees',
        'temporal_resolution': 'annual',
        'references': 'Novel precipitation clustering analysis methodology'
    }
    
    # Save regime data separately
    regime_file = output_dir / f'wd50r_regimes_{year}.txt'
    np.savetxt(regime_file, full_regime, fmt='%s', delimiter=',')
    
    # Save main dataset
    percentiles_str = '_'.join(map(str, percentiles))
    output_file = output_dir / f'wd50r_indices_{year}_P{percentiles_str}.nc'
    
    # Use compression for efficient storage
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} 
               for var in ds.data_vars}
    
    ds.to_netcdf(output_file, encoding=encoding)
    ds.close()
    
    logger.info(f"Saved WD50R indices for {year}: {output_file}")
    
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

def process_year_wd50r(year, precip_dir, output_dir, percentiles, n_processes,
                      chunk_size_lat, chunk_size_lon, wet_day_threshold, min_annual_precip):
    """
    Process WD50R indices for a complete year using parallel processing.
    
    Args:
        year: Target year
        precip_dir: Directory containing precipitation files
        output_dir: Output directory
        percentiles: List of percentiles to calculate
        n_processes: Number of processes for parallel computation
        chunk_size_lat: Latitude chunk size
        chunk_size_lon: Longitude chunk size
        wet_day_threshold: Minimum daily precipitation
        min_annual_precip: Skip pixels below this threshold
        
    Returns:
        Path to output file
    """
    logger.info(f"Processing WD50R indices for year {year} (percentiles: {percentiles})")
    
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
            chunk_args = []
            for chunk_info in spatial_chunks:
                chunk_args.append((chunk_info, year, precip_dir, percentiles,
                                 wet_day_threshold, min_annual_precip))
            
            chunk_results = pool.starmap(process_spatial_chunk_wd50r, chunk_args)
                
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise
    
    # Combine results
    output_file = combine_wd50r_results(chunk_results, full_lat, full_lon, 
                                       year, percentiles, output_dir)
    
    return output_file

def main():
    """Main function for WD50R chronological precipitation indices calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate WD50R chronological precipitation indices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WD50R Methodology:
  For each pixel and year, finds the minimum consecutive days that contain
  the specified percentage of annual precipitation. Provides insights into
  precipitation event clustering and extreme event timing patterns.

Percentile Options:
  25   - WD25R: Very concentrated events
  50   - WD50R: Moderate concentration (standard)
  75   - WD75R: Broader precipitation patterns  
  90   - WD90R: Very distributed patterns

Examples:
  # Standard WD50R analysis
  python 10_WD50R_Chronological_Precipitation_Index.py --year 2020
  
  # Multi-percentile analysis
  python 10_WD50R_Chronological_Precipitation_Index.py --year 2020 --percentiles 25 50 75 90
  
  # Multi-year processing with high performance
  python 10_WD50R_Chronological_Precipitation_Index.py --start-year 2015 --end-year 2020 \\
         --percentiles 25 50 75 --n-processes 48

Computational Notes:
  - Memory: ~3-4 GB per process (intensive sliding window optimization)
  - Processing time: ~1-2 hours per year (depends on percentiles)
  - Early termination optimization significantly improves performance
        """
    )
    
    # Year selection
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument('--year', type=int,
                           help='Process single year')
    year_group.add_argument('--start-year', type=int,
                           help='Start year for multi-year analysis')
    
    parser.add_argument('--end-year', type=int,
                       help='End year for multi-year analysis (required with --start-year)')
    
    # Analysis parameters
    parser.add_argument('--percentiles', nargs='+', type=int, default=[50],
                       help='Percentiles to calculate (default: 50)')
    
    # Data paths
    parser.add_argument('--precip-dir', default='/data/climate/disk1/datasets/era5',
                       help='Directory containing precipitation files')
    parser.add_argument('--output-dir', default='data/processed/wd50r_indices',
                       help='Output directory for WD50R indices')
    
    # Processing parameters
    parser.add_argument('--wet-day-threshold', type=float, default=0.1,
                       help='Minimum daily precipitation (mm/day, default: 0.1)')
    parser.add_argument('--min-annual-precip', type=float, default=5.0,
                       help='Skip pixels below this annual total (mm, default: 50)')
    
    # Computational settings
    parser.add_argument('--n-processes', type=int, default=24,
                       help='Number of processes to use (default: 24)')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size (default: 25, smaller for memory efficiency)')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size (default: 50)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_year and not args.end_year:
        parser.error("--end-year is required when using --start-year")
    
    if not all(1 <= p <= 99 for p in args.percentiles):
        parser.error("Percentiles must be between 1 and 99")
    
    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))
    
    logger.info("="*80)
    logger.info("WD50R CHRONOLOGICAL PRECIPITATION INDICES CALCULATION")
    logger.info("="*80)
    logger.info(f"Analysis years: {years[0] if len(years)==1 else f'{years[0]}-{years[-1]}'}")
    logger.info(f"Percentiles: {args.percentiles}")
    logger.info(f"Precipitation directory: {args.precip_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wet day threshold: {args.wet_day_threshold} mm/day")
    logger.info(f"Minimum annual precipitation: {args.min_annual_precip} mm")
    logger.info(f"Processes: {args.n_processes}")
    logger.info(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    logger.info("="*80)
    
    # Method information
    logger.info("🔍 Analysis Method: Sliding window optimization with early termination")
    logger.info("📊 Output Variables: WDxxR, start_day, end_day, intensity, efficiency")
    logger.info("🏷️  Regime Classification: Based on annual total and clustering patterns")
    
    # Validate directories
    precip_dir = Path(args.precip_dir)
    if not precip_dir.exists():
        raise ValueError(f"Precipitation directory does not exist: {precip_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Estimate computational requirements  
    n_percentiles = len(args.percentiles)
    estimated_memory = 3.5 * args.n_processes  # GB (higher due to sliding window)
    estimated_time = len(years) * n_percentiles * 0.8  # hours
    
    logger.info(f"\n💾 Estimated memory usage: {estimated_memory:.1f} GB")
    logger.info(f"⏱️  Estimated processing time: {estimated_time:.1f} hours")
    logger.info(f"📈 Computational complexity: O(n²) per pixel due to sliding window")
    
    if estimated_memory > 100:
        logger.warning("⚠️  High memory usage - consider reducing --n-processes")
    
    if n_percentiles > 4:
        logger.warning(f"⚠️  Many percentiles ({n_percentiles}) - consider processing in batches")
    
    # Process each year
    successful_years = []
    failed_years = []
    
    for year in years:
        logger.info(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        try:
            output_file = process_year_wd50r(
                year, args.precip_dir, args.output_dir, args.percentiles,
                args.n_processes, args.chunk_size_lat, args.chunk_size_lon,
                args.wet_day_threshold, args.min_annual_precip
            )
            
            if output_file:
                successful_years.append(year)
                logger.info(f"✅ Completed year {year}")
            else:
                failed_years.append(year)
                logger.error(f"❌ Failed to process year {year}")
                
        except Exception as e:
            failed_years.append(year)
            logger.error(f"❌ Error processing year {year}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("WD50R CALCULATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully processed: {len(successful_years)}/{len(years)} years")
    logger.info(f"Percentiles analyzed: {args.percentiles}")
    
    if successful_years:
        logger.info(f"Years completed: {successful_years}")
        logger.info(f"Output directory: {args.output_dir}")
        
        logger.info("\nGenerated files:")
        for year in successful_years[:3]:  # Show first 3
            percentiles_str = '_'.join(map(str, args.percentiles))
            output_file = output_dir / f'wd50r_indices_{year}_P{percentiles_str}.nc'
            if output_file.exists():
                logger.info(f"  - {output_file.name}")
        if len(successful_years) > 3:
            logger.info(f"  ... and {len(successful_years) - 3} more files")
        
        logger.info("\n🎯 WD50R indices calculated successfully!")
        logger.info("⏰ Ready for precipitation timing and clustering analysis!")
        
        # Suggest next steps
        logger.info("\n📋 Next steps:")
        logger.info("1. Create visualization:")
        logger.info(f"   python visualizations/viz_10_wd50r_analysis.py --input-dir {args.output_dir}")
        logger.info("2. Compare with standard WD50:")
        logger.info(f"   python compare_wd50_vs_wd50r.py --wd50r-dir {args.output_dir}")
        logger.info("3. Analyze seasonal patterns:")
        logger.info(f"   python analyze_precipitation_timing.py --input-dir {args.output_dir}")
        
    if failed_years:
        logger.error(f"\n❌ Failed years: {failed_years}")
        logger.error("Check precipitation data availability and computational resources")
    
    if not successful_years:
        logger.error("❌ No years processed successfully!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())