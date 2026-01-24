#!/usr/bin/env python3
"""
Enhanced Precipitation Concentration Indices for Advanced Climate Analysis

Implements advanced precipitation concentration metrics that extend beyond basic WD50:
- PCF: Precipitation Concentration Factor (concentration × abundance)
- EPD: Effective Precipitation Days (multi-threshold analysis: WD10, WD25, WD50, WD75, WD90)
- PGC: Precipitation Gini Coefficient (inequality measure)
- PE: Precipitation Entropy (information-theoretic concentration)
- Regime Classification: Climate-regime specific analysis
- Concentration Slope: Rate of concentration change across thresholds

These indices address limitations of WD50 in arid regions and provide deeper insights
into precipitation pattern changes for climate change analysis.

References:
- Gini coefficient: Damgaard & Weiner (2000) - Ecological inequality measures
- Shannon entropy: Shannon (1948) - Information theory applications
- Concentration curves: Lorenz (1905) - Economic inequality adapted for precipitation
- Multi-threshold analysis: New methodology for precipitation regime characterization

Optimized for high-performance computing with configurable analysis methods.
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

def classify_precipitation_regime(prcptot, r95ptot, wd50):
    """
    Classify precipitation regime based on total precipitation and concentration patterns.
    
    Args:
        prcptot: Annual precipitation from wet days (mm)
        r95ptot: Fraction from extreme days
        wd50: Standard WD50 value
        
    Returns:
        regime: String classification
    """
    if np.isnan(prcptot) or prcptot < 50:
        return "HYPER_ARID"     # Extremely dry
    elif prcptot < 200:
        return "ARID"           # Very dry
    elif prcptot < 500:
        return "SEMI_ARID"      # Moderately dry
    elif not np.isnan(r95ptot) and r95ptot > 0.4:
        return "EXTREME_DOM"    # Extreme-dominated
    elif not np.isnan(wd50) and wd50 < 15:
        return "MONSOON"        # Concentrated wet season
    elif prcptot > 2000:
        return "TROPICAL"       # Very wet
    else:
        return "TEMPERATE"      # Regular patterns

def calculate_concentration_curve_indices(daily_precip, wet_day_threshold=1.0):
    """
    Calculate multi-threshold concentration indices (EPD approach).
    
    Args:
        daily_precip: Daily precipitation array for one year [time]
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        Dictionary with concentration curve indices
    """
    # Filter wet days only
    wet_days = daily_precip[daily_precip >= wet_day_threshold]
    
    if len(wet_days) == 0:
        return {
            'WD10': np.nan, 'WD25': np.nan, 'WD50': np.nan, 
            'WD75': np.nan, 'WD90': np.nan,
            'concentration_slope': np.nan,
            'total_wet_days': 0,
            'effective_precip_days': np.nan
        }
    
    # Sort in descending order
    sorted_precip = np.sort(wet_days)[::-1]
    total_precip = np.sum(sorted_precip)
    
    if total_precip == 0:
        return {
            'WD10': np.nan, 'WD25': np.nan, 'WD50': np.nan, 
            'WD75': np.nan, 'WD90': np.nan,
            'concentration_slope': np.nan,
            'total_wet_days': len(wet_days),
            'effective_precip_days': np.nan
        }
    
    # Calculate cumulative precipitation fractions
    cumulative_precip = np.cumsum(sorted_precip) / total_precip
    
    # Find days for different thresholds
    thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    wd_values = {}
    
    for thresh in thresholds:
        thresh_idx = np.where(cumulative_precip >= thresh)[0]
        if len(thresh_idx) > 0:
            wd_values[f'WD{int(thresh*100)}'] = thresh_idx[0] + 1
        else:
            wd_values[f'WD{int(thresh*100)}'] = len(sorted_precip) if len(sorted_precip) > 0 else np.nan
    
    # Calculate concentration slope (how quickly concentration increases)
    if not np.isnan(wd_values['WD90']) and not np.isnan(wd_values['WD10']):
        concentration_slope = (wd_values['WD90'] - wd_values['WD10']) / 0.8
    else:
        concentration_slope = np.nan
    
    # Effective precipitation days (average of meaningful thresholds)
    meaningful_wd = [wd_values['WD25'], wd_values['WD50'], wd_values['WD75']]
    effective_precip_days = np.mean([wd for wd in meaningful_wd if not np.isnan(wd)])
    
    return {
        'WD10': wd_values['WD10'],
        'WD25': wd_values['WD25'], 
        'WD50': wd_values['WD50'],
        'WD75': wd_values['WD75'],
        'WD90': wd_values['WD90'],
        'concentration_slope': concentration_slope,
        'total_wet_days': len(wet_days),
        'effective_precip_days': effective_precip_days
    }

def calculate_precipitation_gini_coefficient(daily_precip, wet_day_threshold=1.0):
    """
    Calculate Precipitation Gini Coefficient (inequality measure).
    
    Args:
        daily_precip: Daily precipitation array for one year [time]
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        gini_coefficient: Value between 0 (equal) and 1 (concentrated)
    """
    # Filter wet days only
    wet_days = daily_precip[daily_precip >= wet_day_threshold]
    
    if len(wet_days) <= 1:
        return np.nan
    
    # Sort values
    sorted_precip = np.sort(wet_days)
    n = len(sorted_precip)
    
    # Calculate Gini coefficient
    # Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    cumsum = np.cumsum(sorted_precip)
    total_sum = cumsum[-1]
    
    if total_sum == 0:
        return np.nan
    
    # Weighted sum
    weighted_sum = np.sum((np.arange(1, n + 1) * sorted_precip))
    
    gini = (2 * weighted_sum) / (n * total_sum) - (n + 1) / n
    
    return gini

def calculate_precipitation_entropy(daily_precip, wet_day_threshold=1.0):
    """
    Calculate Precipitation Entropy (information-theoretic measure).
    
    Args:
        daily_precip: Daily precipitation array for one year [time]
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        normalized_entropy: Value between 0 (concentrated) and 1 (distributed)
    """
    # Filter wet days only
    wet_days = daily_precip[daily_precip >= wet_day_threshold]
    
    if len(wet_days) <= 1:
        return np.nan
    
    total_precip = np.sum(wet_days)
    
    if total_precip == 0:
        return np.nan
    
    # Convert to probabilities
    precip_probs = wet_days / total_precip
    
    # Calculate Shannon entropy
    # Remove zeros to avoid log(0)
    precip_probs = precip_probs[precip_probs > 0]
    
    if len(precip_probs) <= 1:
        return 0.0  # Complete concentration
    
    entropy = -np.sum(precip_probs * np.log2(precip_probs))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(precip_probs))
    
    if max_entropy == 0:
        return 0.0
    
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

def calculate_precipitation_concentration_factor(prcptot, wd50, regime):
    """
    Calculate Precipitation Concentration Factor (PCF).
    
    Args:
        prcptot: Annual precipitation from wet days (mm)
        wd50: Days for 50% of precipitation
        regime: Precipitation regime classification
        
    Returns:
        pcf: Precipitation Concentration Factor
    """
    if np.isnan(prcptot) or np.isnan(wd50) or prcptot == 0:
        return np.nan
    
    # Base PCF calculation
    concentration_factor = 365 / wd50  # High = concentrated
    abundance_factor = prcptot / 1000  # Normalize by 1000mm
    
    base_pcf = concentration_factor * abundance_factor
    
    # Regime-specific adjustments
    if regime in ["HYPER_ARID", "ARID"]:
        return np.nan  # Not meaningful in very dry regions
    elif regime == "SEMI_ARID":
        # Dampen signal in semi-arid regions
        return base_pcf * 0.5
    elif regime == "EXTREME_DOM":
        # Amplify signal in extreme-dominated regions
        return base_pcf * 1.5
    else:
        return base_pcf

def calculate_enhanced_concentration_indices(precip_data, dates, method="all", wet_day_threshold=1.0):
    """
    Calculate enhanced precipitation concentration indices for a single location and year.
    
    Args:
        precip_data: Daily precipitation array for one year [time]
        dates: Corresponding dates
        method: Analysis method ("basic", "advanced", "all")
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        Dictionary with enhanced concentration indices
    """
    # Handle NaN values
    valid_mask = ~np.isnan(precip_data)
    if np.sum(valid_mask) == 0:
        return {
            'regime': "NO_DATA",
            'PCF': np.nan,
            'WD10': np.nan, 'WD25': np.nan, 'WD50': np.nan, 'WD75': np.nan, 'WD90': np.nan,
            'concentration_slope': np.nan,
            'effective_precip_days': np.nan,
            'PGC': np.nan,
            'PE': np.nan,
            'total_wet_days': 0,
            'PRCPTOT': np.nan,
            'R95pTOT_approx': np.nan
        }
    
    valid_precip = precip_data[valid_mask]
    
    # Additional validation for edge cases
    if len(valid_precip) == 0:
        return {
            'regime': "NO_VALID_DATA",
            'PCF': np.nan,
            'WD10': np.nan, 'WD25': np.nan, 'WD50': np.nan, 'WD75': np.nan, 'WD90': np.nan,
            'concentration_slope': np.nan,
            'effective_precip_days': np.nan,
            'PGC': np.nan,
            'PE': np.nan,
            'total_wet_days': 0,
            'PRCPTOT': np.nan,
            'R95pTOT_approx': np.nan
        }
    
    # Calculate basic metrics needed for regime classification
    wet_days_mask = valid_precip >= wet_day_threshold
    wet_day_precip = valid_precip[wet_days_mask]
    prcptot = np.sum(wet_day_precip)
    
    # Approximate R95pTOT (95th percentile contribution)
    if len(valid_precip) > 10:
        positive_precip = valid_precip[valid_precip > 0]
        if len(positive_precip) > 0:
            p95_threshold = np.percentile(positive_precip, 95)
            r95p = np.sum(valid_precip[valid_precip >= p95_threshold])
            r95ptot_approx = r95p / prcptot if prcptot > 0 else np.nan
        else:
            r95ptot_approx = np.nan
    else:
        r95ptot_approx = np.nan
    
    # Calculate standard WD50 for regime classification
    wd50_basic = np.nan
    if prcptot > 0 and len(valid_precip) > 0:
        sorted_precip = np.sort(valid_precip)[::-1]
        cumulative_precip = np.cumsum(sorted_precip)
        target_precip = prcptot * 0.5
        wd50_idx = np.where(cumulative_precip >= target_precip)[0]
        wd50_basic = wd50_idx[0] + 1 if len(wd50_idx) > 0 else len(valid_precip)
    
    # Classify precipitation regime
    regime = classify_precipitation_regime(prcptot, r95ptot_approx, wd50_basic)
    
    # Initialize results
    results = {
        'regime': regime,
        'PRCPTOT': prcptot,
        'R95pTOT_approx': r95ptot_approx,
        'total_wet_days': len(wet_day_precip)
    }
    
    # Method 1: Basic enhanced indices
    if method in ["basic", "all"]:
        # Concentration curve indices (EPD)
        conc_indices = calculate_concentration_curve_indices(valid_precip, wet_day_threshold)
        results.update(conc_indices)
        
        # Precipitation Concentration Factor (PCF)
        pcf = calculate_precipitation_concentration_factor(prcptot, conc_indices['WD50'], regime)
        results['PCF'] = pcf
    
    # Method 2: Advanced mathematical indices
    if method in ["advanced", "all"]:
        # Precipitation Gini Coefficient
        pgc = calculate_precipitation_gini_coefficient(valid_precip, wet_day_threshold)
        results['PGC'] = pgc
        
        # Precipitation Entropy
        pe = calculate_precipitation_entropy(valid_precip, wet_day_threshold)
        results['PE'] = pe
    
    # Fill missing values for unused methods
    if method == "basic":
        results.update({'PGC': np.nan, 'PE': np.nan})
    elif method == "advanced":
        results.update({
            'PCF': np.nan, 'WD10': np.nan, 'WD25': np.nan, 'WD50': np.nan, 
            'WD75': np.nan, 'WD90': np.nan, 'concentration_slope': np.nan,
            'effective_precip_days': np.nan
        })
    
    return results

def process_spatial_chunk_enhanced(chunk_info, year, precip_dir, method, wet_day_threshold):
    """
    Process enhanced concentration indices for a spatial chunk.
    
    Args:
        chunk_info: (lat_start, lat_end, lon_start, lon_end)
        year: Target year
        precip_dir: Directory containing precipitation files
        method: Analysis method ("basic", "advanced", "all")
        wet_day_threshold: Minimum precipitation for wet day (mm/day)
        
    Returns:
        Dictionary with enhanced concentration index grids for the chunk
    """
    lat_start, lat_end, lon_start, lon_end = chunk_info
    
    logger.info(f"Processing enhanced concentration chunk: year={year}, method={method}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
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
    
    # Define output variables based on method
    if method == "basic":
        var_names = ['regime', 'PCF', 'WD10', 'WD25', 'WD50', 'WD75', 'WD90', 
                    'concentration_slope', 'effective_precip_days', 'total_wet_days', 
                    'PRCPTOT', 'R95pTOT_approx']
    elif method == "advanced":
        var_names = ['regime', 'PGC', 'PE', 'total_wet_days', 'PRCPTOT', 'R95pTOT_approx']
    else:  # method == "all"
        var_names = ['regime', 'PCF', 'WD10', 'WD25', 'WD50', 'WD75', 'WD90',
                    'concentration_slope', 'effective_precip_days', 'PGC', 'PE',
                    'total_wet_days', 'PRCPTOT', 'R95pTOT_approx']
    
    # Initialize output grids
    indices = {}
    for var_name in var_names:
        if var_name == 'regime':
            indices[var_name] = np.full((n_lat, n_lon), "", dtype='U20')
        else:
            indices[var_name] = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    
    # Process each pixel
    for i in range(n_lat):
        for j in range(n_lon):
            pixel_precip = combined_precip[:, i, j]
            
            # Calculate enhanced concentration indices
            pixel_indices = calculate_enhanced_concentration_indices(
                pixel_precip, combined_dates, method, wet_day_threshold
            )
            
            # Store results
            for var_name in var_names:
                if var_name in pixel_indices:
                    indices[var_name][i, j] = pixel_indices[var_name]
    
    logger.info(f"Completed enhanced concentration chunk: year={year}, lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    
    return {
        'chunk_bounds': chunk_info,
        'year': year,
        'method': method,
        'indices': indices
    }

def combine_enhanced_results(chunk_results, full_lat, full_lon, year, method, output_dir):
    """
    Combine spatial chunks into full global arrays and save results.
    
    Args:
        chunk_results: List of chunk processing results
        full_lat: Full latitude array
        full_lon: Full longitude array
        year: Target year
        method: Analysis method used
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
            var_names = list(result['indices'].keys())
            break
    
    if var_names is None:
        logger.error("No successful chunk results to combine")
        return None
    
    # Initialize full grids
    n_lat, n_lon = len(full_lat), len(full_lon)
    
    full_indices = {}
    for var_name in var_names:
        if var_name == 'regime':
            full_indices[var_name] = np.full((n_lat, n_lon), "", dtype='U20')
        else:
            full_indices[var_name] = np.full((n_lat, n_lon), np.nan, dtype=np.float64)
    
    # Fill grids from chunk results
    for result in chunk_results:
        if result is None:
            continue
        
        lat_start, lat_end, lon_start, lon_end = result['chunk_bounds']
        
        for var_name in var_names:
            if var_name in result['indices']:
                full_indices[var_name][lat_start:lat_end, lon_start:lon_end] = result['indices'][var_name]
    
    # Create dataset (exclude string variables from xarray)
    data_vars = {}
    for var_name in var_names:
        if var_name != 'regime':  # Handle regime separately
            data_vars[var_name] = (['latitude', 'longitude'], full_indices[var_name])
    
    ds = xr.Dataset(data_vars, coords={
        'latitude': full_lat,
        'longitude': full_lon
    })
    
    # Add variable attributes
    if 'PCF' in ds.data_vars:
        ds['PCF'].attrs = {
            'long_name': 'Precipitation Concentration Factor',
            'units': 'dimensionless',
            'description': 'Combined measure of precipitation concentration and abundance',
            'formula': '(365/WD50) * (PRCPTOT/1000)',
            'valid_range': [0.0, 50.0],
            'regime_adjusted': 'Yes'
        }
    
    for wd_var in ['WD10', 'WD25', 'WD50', 'WD75', 'WD90']:
        if wd_var in ds.data_vars:
            thresh = int(wd_var[2:])
            ds[wd_var].attrs = {
                'long_name': f'Days for {thresh}% of annual precipitation',
                'units': 'days',
                'description': f'Number of wettest days contributing {thresh}% of annual precipitation',
                'methodology': 'Concentration curve analysis'
            }
    
    if 'concentration_slope' in ds.data_vars:
        ds['concentration_slope'].attrs = {
            'long_name': 'Concentration curve slope',
            'units': 'days per 0.1 fraction',
            'description': 'Rate of concentration change from WD10 to WD90',
            'formula': '(WD90 - WD10) / 0.8'
        }
    
    if 'effective_precip_days' in ds.data_vars:
        ds['effective_precip_days'].attrs = {
            'long_name': 'Effective precipitation days',
            'units': 'days',
            'description': 'Average of WD25, WD50, and WD75',
            'interpretation': 'Representative measure of precipitation concentration'
        }
    
    if 'PGC' in ds.data_vars:
        ds['PGC'].attrs = {
            'long_name': 'Precipitation Gini Coefficient',
            'units': 'dimensionless',
            'description': 'Inequality measure of precipitation distribution',
            'valid_range': [0.0, 1.0],
            'interpretation': '0=equal distribution, 1=complete concentration'
        }
    
    if 'PE' in ds.data_vars:
        ds['PE'].attrs = {
            'long_name': 'Precipitation Entropy',
            'units': 'dimensionless',
            'description': 'Information-theoretic measure of precipitation distribution',
            'valid_range': [0.0, 1.0],
            'interpretation': '0=concentrated, 1=distributed'
        }
    
    if 'PRCPTOT' in ds.data_vars:
        ds['PRCPTOT'].attrs = {
            'long_name': 'Annual precipitation from wet days',
            'units': 'mm',
            'description': 'Total annual precipitation from days >=1mm',
            'wet_day_threshold': '1.0 mm/day'
        }
    
    if 'R95pTOT_approx' in ds.data_vars:
        ds['R95pTOT_approx'].attrs = {
            'long_name': 'Approximate extreme precipitation contribution',
            'units': 'fraction',
            'description': 'Fraction of annual precipitation from days >P95',
            'methodology': 'Annual P95 threshold (approximation)'
        }
    
    # Add global attributes
    ds.attrs = {
        'title': 'Enhanced Precipitation Concentration Indices',
        'description': 'Advanced precipitation concentration metrics beyond standard WD50',
        'source': 'Calculated from ERA5 total precipitation data',
        'year': year,
        'analysis_method': method,
        'wet_day_threshold': '1.0 mm/day',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'creator': 'Enhanced Precipitation Concentration Calculator',
        'methodology': 'Multi-threshold concentration analysis with regime classification',
        'institution': 'Climate Analysis',
        'spatial_resolution': '0.25 degrees',
        'references': 'Gini: Damgaard & Weiner (2000), Entropy: Shannon (1948)'
    }
    
    # Save regime data separately as text file (optional)
    regime_file = output_dir / f'precipitation_regimes_{year}.txt'
    np.savetxt(regime_file, full_indices['regime'], fmt='%s', delimiter=',')
    
    # Save main dataset
    output_file = output_dir / f'enhanced_concentration_indices_{year}_{method}.nc'
    
    # Use compression for efficient storage
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} 
               for var in ds.data_vars}
    
    ds.to_netcdf(output_file, encoding=encoding)
    ds.close()
    
    logger.info(f"Saved enhanced concentration indices for {year} ({method}): {output_file}")
    
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

def process_year_enhanced(year, precip_dir, output_dir, method, n_processes, 
                         chunk_size_lat, chunk_size_lon, wet_day_threshold):
    """
    Process enhanced concentration indices for a complete year using parallel processing.
    
    Args:
        year: Target year
        precip_dir: Directory containing precipitation files
        output_dir: Output directory
        method: Analysis method ("basic", "advanced", "all")
        n_processes: Number of processes for parallel computation
        chunk_size_lat: Latitude chunk size
        chunk_size_lon: Longitude chunk size
        wet_day_threshold: Minimum precipitation for wet day
        
    Returns:
        Path to output file
    """
    logger.info(f"Processing enhanced concentration indices for year {year} (method: {method})")
    
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
                chunk_args.append((chunk_info, year, precip_dir, method, wet_day_threshold))
            
            chunk_results = pool.starmap(process_spatial_chunk_enhanced, chunk_args)
                
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        raise
    
    # Combine results
    output_file = combine_enhanced_results(chunk_results, full_lat, full_lon, 
                                         year, method, output_dir)
    
    return output_file

def main():
    """Main function for enhanced precipitation concentration indices calculation."""
    parser = argparse.ArgumentParser(
        description='Calculate enhanced precipitation concentration indices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Methods:
  basic     - PCF, multi-threshold WD (WD10-90), concentration slope, regime classification
  advanced  - Gini coefficient, precipitation entropy, regime classification  
  all       - All indices combined (recommended for comprehensive analysis)

Examples:
  # Process single year with basic indices
  python 09_Enhanced_Precipitation_Concentration_Indices.py --year 2020 --method basic
  
  # Process multiple years with all indices  
  python 09_Enhanced_Precipitation_Concentration_Indices.py --start-year 2015 --end-year 2020 --method all
  
  # High-performance processing with custom chunking
  python 09_Enhanced_Precipitation_Concentration_Indices.py --start-year 2010 --end-year 2020 \\
         --method all --n-processes 48 --chunk-size-lat 25 --chunk-size-lon 50

Computational Requirements:
  - Memory: ~2-4 GB per process (adjust --n-processes accordingly)
  - Storage: ~500 MB per year per method
  - Recommended: 24-48 processes for optimal performance
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
    
    # Analysis method
    parser.add_argument('--method', choices=['basic', 'advanced', 'all'], default='all',
                       help='Analysis method (default: all)')
    
    # Data paths
    parser.add_argument('--precip-dir', default='/data/climate/disk1/datasets/era5',
                       help='Directory containing precipitation files')
    parser.add_argument('--output-dir', default='data/processed/enhanced_concentration_indices',
                       help='Output directory for enhanced indices')
    
    # Parameters
    parser.add_argument('--wet-day-threshold', type=float, default=1.0,
                       help='Minimum precipitation for wet day (mm/day, default: 1.0)')
    
    # Computational settings
    parser.add_argument('--n-processes', type=int, default=48,
                       help='Number of processes to use (default: 24)')
    parser.add_argument('--chunk-size-lat', type=int, default=50,
                       help='Latitude chunk size (default: 50)')
    parser.add_argument('--chunk-size-lon', type=int, default=100,
                       help='Longitude chunk size (default: 100)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_year and not args.end_year:
        parser.error("--end-year is required when using --start-year")
    
    # Determine years to process
    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))
    
    logger.info("="*80)
    logger.info("ENHANCED PRECIPITATION CONCENTRATION INDICES CALCULATION")
    logger.info("="*80)
    logger.info(f"Analysis years: {years[0] if len(years)==1 else f'{years[0]}-{years[-1]}'}")
    logger.info(f"Analysis method: {args.method}")
    logger.info(f"Precipitation directory: {args.precip_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Wet day threshold: {args.wet_day_threshold} mm/day")
    logger.info(f"Processes: {args.n_processes}")
    logger.info(f"Chunk size: {args.chunk_size_lat} x {args.chunk_size_lon}")
    logger.info("="*80)
    
    # Method-specific information
    if args.method == "basic":
        logger.info("📊 Basic Method: PCF, WD10-90, concentration slope, regime classification")
    elif args.method == "advanced":
        logger.info("🧮 Advanced Method: Gini coefficient, precipitation entropy")
    else:
        logger.info("🎯 Comprehensive Method: All indices for complete analysis")
    
    # Validate directories
    precip_dir = Path(args.precip_dir)
    if not precip_dir.exists():
        raise ValueError(f"Precipitation directory does not exist: {precip_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Estimate computational requirements
    estimated_memory = 2.5 * args.n_processes  # GB
    estimated_time = len(years) * 0.5  # hours (rough estimate)
    
    logger.info(f"\n💾 Estimated memory usage: {estimated_memory:.1f} GB")
    logger.info(f"⏱️  Estimated processing time: {estimated_time:.1f} hours")
    
    if estimated_memory > 100:
        logger.warning("⚠️  High memory usage - consider reducing --n-processes if needed")
    
    # Process each year
    successful_years = []
    failed_years = []
    
    for year in years:
        logger.info(f"\n{'='*20} PROCESSING YEAR {year} {'='*20}")
        
        try:
            output_file = process_year_enhanced(
                year, args.precip_dir, args.output_dir, args.method,
                args.n_processes, args.chunk_size_lat, args.chunk_size_lon,
                args.wet_day_threshold
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
    logger.info("ENHANCED CONCENTRATION CALCULATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Successfully processed: {len(successful_years)}/{len(years)} years")
    logger.info(f"Analysis method: {args.method}")
    
    if successful_years:
        logger.info(f"Years completed: {successful_years}")
        logger.info(f"Output directory: {args.output_dir}")
        
        logger.info("\nGenerated files:")
        for year in successful_years[:5]:  # Show first 5
            output_file = output_dir / f'enhanced_concentration_indices_{year}_{args.method}.nc'
            if output_file.exists():
                logger.info(f"  - {output_file.name}")
        if len(successful_years) > 5:
            logger.info(f"  ... and {len(successful_years) - 5} more files")
        
        logger.info("\n🎯 Enhanced concentration indices calculated successfully!")
        logger.info("📊 Ready for advanced precipitation pattern analysis!")
        
        # Suggest next steps
        logger.info("\n📋 Next steps:")
        logger.info("1. Create visualization:")
        logger.info(f"   python visualizations/viz_09_enhanced_concentration.py --input-dir {args.output_dir}")
        logger.info("2. Analyze trends:")
        logger.info(f"   python analyze_concentration_trends.py --input-dir {args.output_dir}")
        logger.info("3. Compare with ETCCDI:")
        logger.info(f"   python compare_concentration_methods.py --enhanced-dir {args.output_dir}")
        
    if failed_years:
        logger.error(f"\n❌ Failed years: {failed_years}")
        logger.error("Check precipitation data availability and computational resources")
    
    if not successful_years:
        logger.error("❌ No years processed successfully!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())