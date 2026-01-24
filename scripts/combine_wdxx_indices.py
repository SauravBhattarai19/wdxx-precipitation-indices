#!/usr/bin/env python3
"""
Combine Enhanced Precipitation Concentration Indices Files

Reads individual year files and combines WD25, WD50, and WD75 into a single
time-series NetCDF file.

Usage:
    python combine_wdxx_indices.py --input-dir data/processed/enhanced_concentration_indices \
                                   --output-file data/processed/wdxx_1980_2024.nc \
                                   --start-year 1980 --end-year 2024 \
                                   --method all
"""

import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def combine_wdxx_files(input_dir, output_file, start_year, end_year, method='all'):
    """
    Combine individual year files into a single time-series file with WD25, WD50, WD75.
    
    Args:
        input_dir: Directory containing enhanced_concentration_indices_*.nc files
        output_file: Output file path for combined data
        start_year: Start year
        end_year: End year
        method: Method used in original files ('basic', 'advanced', 'all')
        
    Returns:
        Path to output file
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Variables to extract
    target_vars = ['WD25', 'WD50', 'WD75']
    
    # Collect all year files
    year_data = []
    years = []
    
    logger.info(f"Reading files from {input_dir}")
    logger.info(f"Looking for files: enhanced_concentration_indices_*_{method}.nc")
    
    for year in range(start_year, end_year + 1):
        file_path = input_dir / f'enhanced_concentration_indices_{year}_{method}.nc'
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            logger.info(f"Reading {file_path.name}...")
            # Read with decode_timedelta=False to prevent automatic conversion
            ds = xr.open_dataset(file_path, decode_timedelta=False)
            
            # Check if required variables exist
            missing_vars = [var for var in target_vars if var not in ds.data_vars]
            if missing_vars:
                logger.warning(f"Missing variables in {file_path.name}: {missing_vars}")
                ds.close()
                continue
            
            # Extract only WD25, WD50, WD75
            ds_subset = ds[target_vars].copy()
            
            # Convert timedelta64 to numeric days if needed
            for var in target_vars:
                if var in ds_subset.data_vars:
                    if np.issubdtype(ds_subset[var].dtype, np.timedelta64):
                        # Convert timedelta64[ns] to days (float)
                        ds_subset[var] = ds_subset[var] / np.timedelta64(1, 'D')
                        ds_subset[var] = ds_subset[var].astype(np.float32)
            
            ds_subset = ds_subset.assign_coords(year=year)
            
            # Store attributes from first file for later use
            if not year_data:
                # Store original attributes from first file
                for var in target_vars:
                    if var in ds_subset.data_vars:
                        # Keep original attributes
                        pass
            
            year_data.append(ds_subset)
            years.append(year)
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            continue
    
    if not year_data:
        raise ValueError(f"No valid files found in {input_dir} for years {start_year}-{end_year}")
    
    logger.info(f"Successfully read {len(year_data)} files for years {min(years)}-{max(years)}")
    
    # Combine all years into a single dataset
    logger.info("Combining years into single dataset...")
    
    # Concatenate along a new time dimension
    combined = xr.concat(year_data, dim='year')
    
    # Convert timedelta64 to numeric days if still present after concatenation
    for var in target_vars:
        if var in combined.data_vars:
            if np.issubdtype(combined[var].dtype, np.timedelta64):
                # Convert timedelta64[ns] to days (float)
                # Get the numpy array, convert, and create new DataArray
                values = combined[var].values
                if np.issubdtype(values.dtype, np.timedelta64):
                    days_values = values / np.timedelta64(1, 'D')
                    combined[var] = xr.DataArray(
                        days_values.astype(np.float32),
                        dims=combined[var].dims,
                        coords=combined[var].coords,
                        attrs=combined[var].attrs
                    )
    
    # Rename 'year' coordinate to 'time' and convert to datetime
    # Create time coordinates as January 1st of each year
    time_coords = [datetime(year, 1, 1) for year in years]
    combined = combined.rename({'year': 'time'})
    combined = combined.assign_coords(time=time_coords)
    
    # Attributes are already preserved from original files, no need to modify
    
    # Add global attributes
    combined.attrs = {
        'title': 'Precipitation Concentration Indices (WD25, WD50, WD75)',
        'description': 'Combined time-series of precipitation concentration indices',
        'source': 'Calculated from ERA5 total precipitation data',
        'years': f'{min(years)}-{max(years)}',
        'variables': 'WD25, WD50, WD75',
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'creator': 'WDXX Index Combiner',
        'methodology': 'Multi-threshold concentration analysis',
        'institution': 'Climate Analysis',
        'spatial_resolution': '0.25 degrees'
    }
    
    # Save to file
    logger.info(f"Saving combined data to {output_file}...")
    
    # Clear any existing encoding to avoid conflicts with attrs
    for var in combined.data_vars:
        combined[var].encoding.clear()
    
    # Use compression for efficient storage
    encoding = {var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} 
               for var in combined.data_vars}
    
    combined.to_netcdf(output_file, encoding=encoding)
    combined.close()
    
    logger.info(f"✅ Successfully created {output_file}")
    logger.info(f"   Variables: {', '.join(target_vars)}")
    logger.info(f"   Years: {min(years)}-{max(years)} ({len(years)} years)")
    logger.info(f"   Dimensions: {dict(combined.dims)}")
    
    return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Combine enhanced precipitation concentration indices files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine files from default location
  python combine_wdxx_indices.py --start-year 1980 --end-year 2024
  
  # Specify custom input/output directories
  python combine_wdxx_indices.py --input-dir data/processed/enhanced_concentration_indices \\
                                 --output-file data/processed/wdxx_1980_2024.nc \\
                                 --start-year 1980 --end-year 2024 --method all
        """
    )
    
    parser.add_argument('--input-dir', 
                       default='data/processed/enhanced_concentration_indices',
                       help='Directory containing enhanced_concentration_indices_*.nc files')
    parser.add_argument('--output-file',
                       default='data/processed/wdxx_1980_2024.nc',
                       help='Output file path for combined data')
    parser.add_argument('--start-year', type=int, default=1980,
                       help='Start year (default: 1980)')
    parser.add_argument('--end-year', type=int, default=2024,
                       help='End year (default: 2024)')
    parser.add_argument('--method', choices=['basic', 'advanced', 'all'], 
                       default='all',
                       help='Method used in original files (default: all)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COMBINE WDXX INDICES")
    logger.info("="*80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Years: {args.start_year}-{args.end_year}")
    logger.info(f"Method: {args.method}")
    logger.info("="*80)
    
    try:
        output_file = combine_wdxx_files(
            args.input_dir,
            args.output_file,
            args.start_year,
            args.end_year,
            args.method
        )
        
        logger.info("\n" + "="*80)
        logger.info("SUCCESS!")
        logger.info("="*80)
        logger.info(f"Combined file created: {output_file}")
        logger.info(f"Variables: WD25, WD50, WD75")
        logger.info(f"Time range: {args.start_year}-{args.end_year}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
