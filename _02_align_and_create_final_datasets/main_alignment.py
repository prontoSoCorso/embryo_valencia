#!/usr/bin/env python3
"""
preprocess_and_merge_time_series.py

Pipeline:
 - Load 'timings.pkl' (contains list of timestamps for each video).
 - Load the chosen metric pickle (e.g., 'vorticity.pkl', 'sum_mean_mag.pkl').
 - Load the Reference CSV (metadata).
 - For each embryo:
     * Retrieve values and specific timings.
     * Smooth (optional, configurable).
     * Resample/Interpolate onto a regular 15-min grid (0.00h, 0.25h, ...).
 - Save the processed pickle (aligned data).
 - Build and save the Final CSV (Metadata + Aligned Time Series).
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

# Add parent directory to sys.path to import config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import Config_OpticalFlow as config

# ---------------- CONFIGURATION ----------------

# Input Pickles
PICKLE_DIR              = config.pickle_dir
TIMINGS_PICKLE_NAME     = "timings.pkl"
# CHANGE THIS to "sum_mean_mag.pkl", "mean_magnitude.pkl", or "vorticity.pkl"
CHOSEN_METRIC_NAME      = "sum_mean_mag.pkl"   

# Reference Metadata CSV
# Assuming the metadata file is the one used previously for sorting videos
REFERENCE_CSV           = "/home/phd2/Scrivania/CorsoRepo/embryo_valencia/datasets/dataset_final_merged.csv"

# Output Paths
OUT_PICKLE_DIR          = os.path.join(config.base_project_database_path, "processed_pickles")
FINAL_CSV_DIR           = os.path.join(config.base_project_database_path, "final_datasets")

# Processing Options
GRID_STEP_HOURS             = 0.25      # 15 minutes
SMOOTH_METHOD               = 'savgol'  # 'savgol', 'median', or None
SMOOTH_WINDOW               = 5         # Must be odd for savgol (e.g., 5 or 7)
SMOOTH_POLYORDER            = 3
TRUNCATE_TO_GRID_EXTENT     = False     # If True, truncates values outside grid range
TRIMMING_MAX_LENGTH         = 528       # Max time points to keep (approx 5.5 days)

# Create output directories
os.makedirs(OUT_PICKLE_DIR, exist_ok=True)
os.makedirs(FINAL_CSV_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
try:
    from scipy.signal import savgol_filter
    _has_savgol = True
except Exception:
    _has_savgol = False

def load_pickle_file(path):
    """Safely loads a pickle file."""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded {os.path.basename(path)}: {len(data)} entries found.")
        return data
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None

def load_csv(csv_path):
    """Loads CSV, handling delimiter sniffing if possible."""
    try:
        return pd.read_csv(csv_path, sep=None, engine='python')
    except Exception as e:
        print(f"ERROR loading CSV {csv_path}: {e}")
        return pd.DataFrame()

def smooth_array(a, method='savgol', window=5, polyorder=3):
    """Applies smoothing to the array."""
    a = np.asarray(a, dtype=float)
    if method is None:
        return a
    
    # Savitzky-Golay filter
    if method == 'savgol' and _has_savgol and (window is not None):
        if window % 2 == 0: window += 1 # Ensure odd
        if len(a) >= window:
            try:
                return savgol_filter(a, window_length=window, polyorder=polyorder)
            except Exception:
                pass
    
    # Simple Moving Average / Median fallback
    if method in ('median', 'savgol'):
        k = 3
        if len(a) < k: return a
        kernel = np.ones(k) / k
        padded = np.pad(a, (k//2, k//2), mode='edge')
        smooth = np.convolve(padded, kernel, mode='valid')
        return smooth[:len(a)]
        
    return a

def resample_to_grid(values, original_times, grid_hours):
    """
    Resamples irregular time series (values, original_times) onto a regular grid (grid_hours).
    Uses linear interpolation.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(original_times, dtype=float)
    
    # Handle empty or mismatch
    if len(times) == 0 or len(values) == 0:
        return np.full(len(grid_hours), np.nan)
        
    # Align lengths (truncate to shortest)
    n = min(len(times), len(values))
    times = times[:n]
    values = values[:n]
    
    # Sort by time to ensure monotonicity for interpolation
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    
    # Linear Interpolation
    # left=NaN, right=NaN ensures we don't extrapolate blindly
    res = np.interp(grid_hours, times, values, left=np.nan, right=np.nan)
    
    return res

# ---------------- MAIN PIPELINE ----------------

def main():
    print("--- Starting Preprocessing Pipeline ---")
    
    # 1. Load Data
    # Load Timings
    timings_path = os.path.join(PICKLE_DIR, TIMINGS_PICKLE_NAME)
    timings_data = load_pickle_file(timings_path)
    
    # Load Chosen Metric
    metric_path = os.path.join(PICKLE_DIR, CHOSEN_METRIC_NAME)
    metric_data = load_pickle_file(metric_path)
    
    if not timings_data or not metric_data:
        print("Missing input pickles. Exiting.")
        return

    # Load Metadata CSV
    csv_data = load_csv(REFERENCE_CSV)
    if csv_data.empty:
        print("Reference CSV empty or not found. Exiting.")
        return

    # 2. Clean Metadata (Dedup)
    if 'dish_well' in csv_data.columns:
        csv_data['dish_well'] = csv_data['dish_well'].astype(str).str.strip()
        before = len(csv_data)
        csv_data = csv_data.drop_duplicates(subset='dish_well', keep='first')
        if len(csv_data) < before:
            print(f"Removed {before - len(csv_data)} duplicate rows from metadata.")
    else:
        print("WARNING: 'dish_well' column not found in reference CSV.")

    # 3. Define Master Grid
    # Find the maximum time across all videos to determine grid size
    all_times = np.concatenate(list(timings_data.values())) if timings_data else np.array([])
    max_h = float(np.nanmax(all_times)) if len(all_times) > 0 else 0.0
    
    # Cap max duration if needed
    limit_h = TRIMMING_MAX_LENGTH * GRID_STEP_HOURS
    if max_h > limit_h: max_h = limit_h
    
    grid_hours = np.arange(0.0, max_h + GRID_STEP_HOURS, GRID_STEP_HOURS)
    grid_col_names = [f"{h:.2f}h" for h in grid_hours]
    print(f"Grid defined: 0.00h to {max_h:.2f}h ({len(grid_hours)} steps)")

    # 4. Process Each Series
    processed_data = {}
    
    print(f"Processing metric: {CHOSEN_METRIC_NAME}...")
    
    for dish_well, series_raw in metric_data.items():
        # Get timings
        times_raw = timings_data.get(dish_well)
        
        if times_raw is None:
            # Fallback: create synthetic time if timing is missing
            # logging.warning(f"No timings for {dish_well}, using regular steps.")
            times_raw = np.arange(len(series_raw)) * GRID_STEP_HOURS
        
        # Smooth (skip smoothing if it's sum_mean_mag as it is already an aggregate)
        vals = np.array(series_raw, dtype=float)
        
        if not CHOSEN_METRIC_NAME == "" and SMOOTH_METHOD:
            vals = smooth_array(vals, method=SMOOTH_METHOD, window=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER)
            
        # Resample to Grid
        vals_aligned = resample_to_grid(vals, times_raw, grid_hours)
        
        processed_data[dish_well] = vals_aligned

    # 5. Save Processed Pickle (Aligned Data)
    proc_pickle_path = os.path.join(OUT_PICKLE_DIR, f"aligned_{CHOSEN_METRIC_NAME}")
    with open(proc_pickle_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved aligned pickle to: {proc_pickle_path}")

    # 6. Build Final CSV
    final_rows = []
    
    # We iterate over the CSV rows to keep metadata structure
    for _, row in csv_data.iterrows():
        dish_id = str(row['dish_well'])
        
        # Start with metadata fields
        new_row = row.to_dict()
        
        # Get metric data (or NaNs if missing)
        if dish_id in processed_data:
            metric_vals = processed_data[dish_id]
        else:
            metric_vals = np.full(len(grid_hours), np.nan)
            
        # Append time columns
        for t_col, val in zip(grid_col_names, metric_vals):
            # Format nicely (optional, helps CSV readability)
            new_row[t_col] = f"{val:.5f}" if not np.isnan(val) else ""
            
        final_rows.append(new_row)

    final_df = pd.DataFrame(final_rows)
    
    # Construct Filename
    metric_tag = CHOSEN_METRIC_NAME.replace(".pkl", "")
    final_csv_name = f"FinalDataset_{metric_tag}.csv"
    final_csv_path = os.path.join(FINAL_CSV_DIR, final_csv_name)
    
    # Save (Semicolon sep for Europe/Italy)
    final_df.to_csv(final_csv_path, index=False, sep=';')
    print(f"Saved Final CSV to: {final_csv_path}")
    print(f"Total Rows: {len(final_df)}")

if __name__ == '__main__':
    main()